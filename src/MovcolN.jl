module MovcolN

using Polynomials
using Devectorize

export getcomputeres

include("polynomials.jl")       # standard polynomial related
                                # functions

# create matrix AB=inv(A)*B with A and B such that A*Fi = B*Gi/Hi
# (i.e. approximating F=∂ₓG), assuming that Fi is evaluated at Gauss
# points and Gi is evaluated at Lobatto points.  The dimensions of A
# and B are n×n and n×n+1 respectively.
function generateAB(n; T=Float64)
    sg, _ = gauss(n)
    sl, _ = lobatto(n+1)
    x    = Poly(T[0,1])
    Lint = polyint(Lagrange(sg,x))
    # A = ∫Li dx over the interval [sl[j],sl[j+1]]
    A  = -T[polyval(Lint[k],sl[j+1])-polyval(Lint[k],sl[j])
           for j=1:n, k=1:n]
    B  = spdiagm((fill(1,n),fill(-1,n)),(0,1))
    AB = inv(A)*B
    return AB
end


# generates an array of matrices Q such that
# HiᵏUₓ⁽ᵏ⁾(xi)=Q[k+1]*(H.*U12), where U12 is a vector
# [u1,u1x,u1xx,...,u2,u2x,u2xx,...] of length 2(n+1),
# H=[1,Hi,Hi^2,...,Hi^n,1,Hi,Hi^2,...,Hi^n] and xi are fixed points
# (in practice, these are either Gauss or Lobatto points)
#
# @todo this wont work for integer xi, because it will try to convert
# Rational into integer
function generateQatx{T}(n,x::T;kmax=n)
    L = hermite01(n)
    Qleft  = T[polyval(polyder(L[1,j],i),x) for i=0:kmax-1, j=1:n]
    Qright = T[polyval(polyder(L[2,j],i),x) for i=0:kmax-1, j=1:n]
    return Qleft, Qright, x
end


function computeux!{T}(ux::Array{T,2},Qs,H::Vector{T},ul::Matrix{T},ur::Matrix{T})
    Qleft, Qright, _ = Qs
    nd, nu = size(ul)
    nq = size(Qleft,1)

    Hul = Array(T,nd)
    Hur = Array(T,nd)

    for i=1:nu
        @devec Hul[:] = H[1:nd].*ul[:,i]
        @devec Hur[:] = H[1:nd].*ur[:,i]
        ux[:,i] = Qleft*Hul + Qright*Hur
        @devec ux[:,i] = ux[:,i]./H[1:nq]
    end
end


function computeutx!{T}(utx::Array{T,2},Qs,H,Ht,xt,utl,utr,ux,ul,ur)
    Qleft, Qright, s = Qs
    nd, nu = size(utl)
    h  = H[2]
    ht = Ht[2]

    Hul = Array(T,nd)
    Hur = Array(T,nd)

    for i = 1:nu
        @devec Hul[:] = H[1:nd].*utl[:,i]+Ht[1:nd].*ul[:,i]
        @devec Hur[:] = H[1:nd].*utr[:,i]+Ht[1:nd].*ur[:,i]
        utx[:,i] = Qleft*Hul+Qright*Hur
        xt = (xt+s*ht)
        # upscale spatial derivatives
        @devec utx[:,i] = utx[:,i]./H
        for j = 1:size(utx,1)-1
            # correct for the dilation and shift of the mesh
            utx[j,i] += -((j-1)*ht/h*ux[j,i]+xt*ux[j+1,i]) # j*ht*h^(j-1) = d/dt(h^j)
        end
    end
end

function getcollocationvalues!(FVal,Qs,F,t,H,Ht,x,xt,ul,ur,utl,utr)
    h = H[2]                    # H[2] = h^1 = h
    T   = eltype(t)
    nq  = size(Qs[1][1],1)      # @todo get rid of nq and nu
                                # altogether
    nu  = size(ul,2)
    ux  = Array(T,nq,nu)
    utx = zero(ux)

    for j = 1:length(Qs)
        _,_,s  = Qs[j]
         computeux!( ux,Qs[j],H,ul,ur)
        computeutx!(utx,Qs[j],H,Ht,xt,utl,utr,ux,ul,ur)
        xs  = x+s*h
        FVal[:,j] = F(t,xs,ux,utx)
    end
end


# memory alignment
#
#      derivative (nd)
#     /   point (nx)
#    /   /
# u[i,j,k]
#     \
#      function (nu)
#
# generates the residual function of the equation eqn

function getcomputeres(F,G,Bl,Br,M,Bxl,Bxr,tau,gamma,
                       ns,      # number of derivatives to hold
                       nu,      # number of dependent variables
                       nx)      # number of mesh points

    if !(iseven(ns))
        error("ns should be even")
    end

    T = Float64

    kmax = ns+1                 # maximal darivative is of rank kmax-1 = ns
    ns2 = Int(ns/2)
    sGauss, _ = gauss(ns)       # Gauss points
    sLobat, _ = lobatto(ns+1)   # Lobatto points

    QsGauss = map(x->generateQatx(ns,x,kmax=kmax),sGauss)
    QsLobat = map(x->generateQatx(ns,x,kmax=kmax),sLobat)

    FatGauss = Array(T,nu,ns  )
    GatLobat = Array(T,nu,ns+1)

    resu     = Array(T,nu,ns,  nx)
    resx     = Array(T,        nx)
    res      = Array(T,ns*nu*nx+nx)
    Mhalf    = Array(T,        nx-1)
    Yhalf    = Array(T,        nx+1)
    uhalf    = Array(T,ns+1,nu)
    uthalf   = Array(T,ns+1,nu)

    AB = generateAB(ns)

    function computeres{T}(t :: T,
                           x :: Vector{T}, xt :: Vector{T},
                           u :: Array{T,3},ut :: Array{T,3})

        # @todo xt = g(t)*xt, ut = g(t)*ut and make tau=tau(t)

        for i = 1:nx-1
            # @todo create an immutable with h,ht,ul,ur,utl,utr?
             xl =  x[i];      xr =  x[i+1]
            xtl = xt[i];     xtr = xt[i+1]
             ul =  u[:,:,i];  ur =  u[:,:,i+1]
            utl = ut[:,:,i]; utr = ut[:,:,i+1]
            h  =  xr-xl; ht = xtr-xtl
            H  = [h^j for j=0:ns]
            Ht = [j*ht*h^(j-1) for j=0:ns]

            #######################################################################
            # pde residua
            #######################################################################
            getcollocationvalues!(FatGauss,QsGauss,F,t,H,Ht,xl,xtl,ul,ur,utl,utr)
            getcollocationvalues!(GatLobat,QsLobat,G,t,H,Ht,xl,xtl,ul,ur,utl,utr)

            # @todo change the definition of AB or FatGauss to remove
            # the primes
            resu[:,:,i] = FatGauss'-AB*GatLobat'/h
            # residua for the boundary conditions, Bl and Br should each
            # return the vector of size at least nu*ns/2, the remaining
            # boundary conditions are discarded
            if i == 1
                resBl = Bl(t,xl,xtl,ul,utl)
                resu[:,    1:ns2,nx] = reshape(resBl[1:nu*ns2],nu,ns2)
            elseif i == nx-1
                resBr = Br(t,xr,xtr,ur,utr)
                resu[:,ns2+1:ns, nx] = reshape(resBr[1:nu*ns2],nu,ns2)
            end

            #######################################################################
            # mesh movement residua
            #######################################################################
            if i==1 || i==nx-1
                Mhalf[i] = (M(t,xl,ul,utl)+M(t,xr,ur,utr))/2
            else
                xhalf  =  xl+ h/2
                xthalf = xtl+ht/2
                 computeux!(uhalf, QsLobat[ns2+1],H,ul,ur)
                computeutx!(uthalf,QsLobat[ns2+1],H,Ht,xthalf,utl,utr,uhalf,ul,ur)
                Mhalf[i] = M(t,xhalf,uhalf,uthalf)
            end

            Yhalf[i+1] = -tau*(xtr-xtl)/(xr-xl)^2+1/(xr-xl)

            if i == 1
                resx[ 1] = Bxl(t,xl,xtl,ul,utl)
            elseif i == nx-1
                resx[nx] = Bxr(t,xr,xtr,ur,utr)
            end
        end

        # boundary conditions for Yhalf
        Yhalf[ 1 ] = Yhalf[  2  ]
        Yhalf[end] = Yhalf[end-1]

        # compute the residua for mesh movement
        for i = 2:nx-1
            g=gamma*(gamma+1)
            resx[i] = (Yhalf[i+1]-g*(Yhalf[i+2]-2*Yhalf[i+1]+Yhalf[i]))/Mhalf[i]-(Yhalf[i]-g*(Yhalf[i+1]-2*Yhalf[i]+Yhalf[i-1]))/Mhalf[i-1]
        end

        return copy(resu), copy(resx)

    end

    function daewrapper(t,y,yt)
         x =  y[1:nx];  u = reshape( y[nx+1:end],ns,nu,nx)
        xt = yt[1:nx]; ut = reshape(yt[nx+1:end],ns,nu,nx)
        resu,resx = computeres(t,x,xt,u,ut)
        res[1:nx] = resx
        res[nx+1:end] = resu
        return copy(res)
    end

    return computeres, daewrapper

end

end # module
