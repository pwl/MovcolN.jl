module MovcolN

using Polynomials
using Devectorize
using DASSL
using ArrayViews

include("polynomials.jl")       # polynomial related functions

export Equation, movcol_solve

# Equation should consist of functions F,G,Bl,Br,M,g,tau,Bxl,Bxr,gamma
abstract Equation

# CollocationData is the same for every mesh point and it depends on
# the number of collocation points (ns) only.
immutable CollocationData # only works with Float64
    ns        # number of derivatives provided at mesh points

    # vectors of tuples (Qleft, Qright, s)
    gauss   :: Vector{(Matrix,Matrix,Real)}
    lobatto :: Vector{(Matrix,Matrix,Real)}

    AB :: Matrix
end


# @todo tests...
function CollocationData(ns)
    QsGauss   = map(s->generateQats(ns,s),  gauss(ns)[1])
    QsLobatto = map(s->generateQats(ns,s),lobatto(ns+1)[1])
    AB = generateAB(ns)
    return CollocationData(ns,QsGauss,QsLobatto,AB)
end


immutable MeshPair{T}
    left  :: (T,T,AbstractArray{T,2},AbstractArray{T,2})
    right :: (T,T,AbstractArray{T,2},AbstractArray{T,2})
    h     :: T
    ht    :: T
    H     :: Array{T,1}
    Ht    :: Array{T,1}
    xt    :: T
    nd    :: Int
    nu    :: Int
end

# @todo tests
function MeshPair{T}(x::Vector{T},xt::Vector{T},u::Array{T,3},ut::Array{T,3},i::Int)
    nd,nu,_ = size(u)
    left  = (x[i  ],xt[i  ],u[:,:,i  ],ut[:,:,i  ])
    right = (x[i+1],xt[i+1],u[:,:,i+1],ut[:,:,i+1])
    left  = (x[i  ],xt[i  ],view(u,:,:,i  ),view(ut,:,:,i  ))
    right = (x[i+1],xt[i+1],view(u,:,:,i+1),view(ut,:,:,i+1))
    h = x[i+1]-x[i];      ht = xt[i+1]-xt[i]
    H  = [h^j for j=0:nd]
    Ht = [j*ht*h^(j-1) for j=0:nd]
    return MeshPair(left,right,h,ht,H,Ht,xt[i],nd,nu)
end


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
function generateQats{T}(n,s::T;kmax=n+1)
    L = hermite01(n)
    Qleft  = T[polyval(polyder(L[1,j],i),s) for i=0:kmax-1, j=1:n]
    Qright = T[polyval(polyder(L[2,j],i),s) for i=0:kmax-1, j=1:n]
    return Qleft, Qright, s
end


# given the vectors ul=[u,ux,uxx,...]_(x=x[N]) and
# ur=[u,ux,uxx,...]_(x=x[N+1]) generate a vector
# ux=[u,ux,uxx,...]_(x=x[N]+s*h), where s is prescribed in Qs (this
# descriptions of ul, ur and ux are oversimplification as each of them
# holds the derivatives of more than just one function,
# e.g. ul[j,i]=∂ⱼuᵢ(x[N]) ).  Generation of ux is realized by
# performing linear operations on ul and ur via the matrices Qleft and
# Qright.  The vector H is equal to [h^j for j=0:nd+1].
function computeux!{T}(ux::AbstractArray{T,2},Qs,H::AbstractVector{T},ul::AbstractMatrix{T},ur::AbstractMatrix{T})
    Qleft, Qright, s = Qs
    nd, nu = size(ul)

    Hul = Array(T,nd)
    Hur = Array(T,nd)

    for i=1:nu
        @devec Hul[:] = H[1:nd].*ul[:,i]
        @devec Hur[:] = H[1:nd].*ur[:,i]
        ux[:,i] = Qleft*Hul + Qright*Hur
        @devec ux[:,i] = ux[:,i]./H
    end
end


# Function similar to computeux! but computing
# utx=[ut,utx,utxx,...]. It requires more data (Ht,xt,utl,utr,ux)
# because it has to account for the mesh translation and rescaling in
# time.
function computeutx!{T}(utx::AbstractArray{T,2},Qs,H,Ht,xt,utl,utr,ux,ul,ur)
    Qleft, Qright, s = Qs
    nd, nu = size(utl)
    h  = H[2]
    ht = Ht[2]
    xt = (xt+s*ht)

    Hul = Array(T,nd)
    Hur = Array(T,nd)

    for i = 1:nu
        @devec Hul[:] = H[1:nd].*utl[:,i]+Ht[1:nd].*ul[:,i]
        @devec Hur[:] = H[1:nd].*utr[:,i]+Ht[1:nd].*ur[:,i]
        utx[:,i] = Qleft*Hul+Qright*Hur
        # upscale spatial derivatives
        @devec utx[:,i] = utx[:,i]./H
        for j = 1:size(utx,1)-1
            # correct for the dilation and shift of the mesh
            utx[j,i] += -((j-1)*ht/h*ux[j,i]+xt*ux[j+1,i]) # j*ht*h^(j-1) = d/dt(h^j)
        end
    end
end

# For a given function F(t,u,ux,uxx,...,ut,utx,utxx,...) return the
# value F_(x=x[N]+s*h).  The spatial and temporal derivatives at
# x=x[N]+s*h are computed via computeux! and computeutx! functions.
function getcollocationvalues!{T}(FVal,Qs,F,t::T,mp::MeshPair{T})
    x,xt,ul,utl = mp.left
    _,_ ,ur,utr = mp.right
    H,Ht = mp.H,mp.Ht
    ux  = Array(T,mp.nd+1,mp.nu)
    utx = Array(T,mp.nd+1,mp.nu)

    for j = 1:length(Qs)
        _,_,s  = Qs[j]
         computeux!(ux, Qs[j],H,ul,ur)
        computeutx!(utx,Qs[j],H,Ht,xt,utl,utr,ux,ul,ur)
        xs  = x+s*mp.h
        FVal[:,j] = F(t,xs,ux,utx)
    end
end


# memory alignment
#
#      derivative (nd)
#     /   point (nx)
#    /   /
# u[i,j,k]
#      \
#       function (nu)
#

function computeresu{T}(pde :: Equation,
                        coldata :: CollocationData,
                        t  :: T,
                        x  :: Vector{T}, xt :: Vector{T},
                        u  :: Array{T,3},ut :: Array{T,3})
    nd,nu,nx = size(u)
    ns = nd
    ns2 = int(ns/2)

    FGauss   = Array(T,nu,ns  )
    GLobatto = Array(T,nu,ns+1)
    resu     = Array(T,nu,ns,nx)

    for i = 1:nx-1
        mp = MeshPair(x,xt,u,ut,i)

        getcollocationvalues!(FGauss,  coldata.gauss,  pde.F,t,mp)
        getcollocationvalues!(GLobatto,coldata.lobatto,pde.G,t,mp)

        # @todo change the definition of AB or FGauss to remove the
        # primes
        resu[:,:,i] = FGauss'-coldata.AB*GLobatto'/mp.h
    end

    # residua for the boundary conditions, Bl and Br should each
    # return the vector of size at least nu*ns/2, the remaining
    # boundary conditions are discarded
    mp1  = MeshPair(x,xt,u,ut,1 )
    mpnx = MeshPair(x,xt,u,ut,nx-1)
    resBl = pde.Bl(t,mp1.left...)
    resBr = pde.Bl(t,mpnx.right...)
    resu[:,    1:ns2,nx] = reshape(resBl[1:nu*ns2],nu,ns2)
    resu[:,ns2+1:ns, nx] = reshape(resBr[1:nu*ns2],nu,ns2)

    return resu

end


# compute residua for the mesh
function computeresx{T}(pde :: Equation,
                        coldata :: CollocationData,
                        t :: T,
                        x :: Vector{T}, xt :: Vector{T},
                        u :: Array{T,3},ut :: Array{T,3})

    nd,nu,nx = size(u)
    ns = nd                     # number of collocation points
    ns2 = int(ns/2)

    uhalf    = Array(T,ns+1,nu)
    uthalf   = Array(T,ns+1,nu)
    resx     = Array(T,nx)
    Mhalf    = Array(T,nx-1)
    Yhalf    = Array(T,nx+1)

    Qshalf = coldata.lobatto[ns2+1]

    for i = 1:nx-1
        mp = MeshPair(x,xt,u,ut,i)
        xl,xtl,ul,utl = mp.left
        xr,xtr,ur,utr = mp.right
        h, ht, H, Ht = mp.h, mp.ht, mp.H, mp.Ht

        if i==1 || i==nx-1
            Mhalf[i] = (pde.M(t,xl,ul)+pde.M(t,xr,ur))/2
        else
            xhalf  =  xl+ h/2
            xthalf = xtl+ht/2
             computeux!(uhalf, Qshalf,H,ul,ur)
            computeutx!(uthalf,Qshalf,H,Ht,xthalf,utl,utr,uhalf,ul,ur)
            Mhalf[i] = pde.M(t,xhalf,uhalf)
        end

        Yhalf[i+1] = -pde.tau*(xtr-xtl)/(xr-xl)^2+1/(xr-xl)

        if i == 1
            resx[ 1] = pde.Bxl(t,xl,xtl,ul,utl)
        end
        if i == nx-1
            resx[nx] = pde.Bxr(t,xr,xtr,ur,utr)
        end
    end

    # boundary conditions for Yhalf
    Yhalf[ 1 ] = Yhalf[  2  ]
    Yhalf[end] = Yhalf[end-1]

    # @todo add monitor smoothing

    # compute the residua for mesh movement
    for i = 2:nx-1
        g=pde.gamma*(pde.gamma+1)
        resx[i] = (Yhalf[i+1]-g*(Yhalf[i+2]-2*Yhalf[i+1]+Yhalf[i]))/Mhalf[i]-(Yhalf[i]-g*(Yhalf[i+1]-2*Yhalf[i]+Yhalf[i-1]))/Mhalf[i-1]
    end

    return resx

end


# creates a relaxed mesh
function meshinit(pde :: Equation,
                  coldata :: CollocationData,
                  nx :: Int;
                  mesherr = 1e-5,
                  maxsteps = 1000,
                  args...)

    ns = coldata.ns
    x0 = linspace(pde.xspan...,nx)
    x0t = zero(x0)

    # Create a new pde with fixed mesh endpoints
    xpde = deepcopy(pde)
    xpde.Bxl=(t,x,xt,u,ut)->xt
    xpde.Bxr=(t,x,xt,u,ut)->xt

    # wrapper function for DASSL
    function dae(t,x,xt)
        u    = reshape(vcat(map(xpde.u0, x)...),ns,1,nx)
        resx = computeresx(xpde,coldata,xpde.t0,x,xt,u,u)
        return resx
    end

    nstp = 0
    for (t,x,xt) in dasslIterator(dae,x0,0.0;reltol=mesherr*1e-2,abstol=mesherr*1e-2)
        nstp += 1

        if norm(xt,Inf) < mesherr
            info("Mesh converged after $nstp steps")
            x0  = x
            break
        elseif nstp > maxsteps
            error("Unable to converge in $maxsteps steps")
            break
        end
    end

    return x0

end

function movcol_solve(pde :: Equation,
                      nx :: Int,
                      nd :: Int;
                      args...)

    coldata = CollocationData(nd)
    if isodd(nd); error("nd should be even"); end
    if nx < 2; error("nx should be at least 2"); end
    nu = pde.nu
    x0 = meshinit(pde,coldata,nx;args...)
    u0 = reshape(vcat(map(pde.u0, x0)...),nd,nu,nx)

    # wrapper function for DASSL
    function dae(t,y,yt)
        res = Array(eltype(x0),nd*nu*nx+nx)
        # @todo xt = g(t)*xt, ut = g(t)*ut and make tau=tau(t)
         x =  y[1:nx];  u = reshape( y[nx+1:end],nd,nu,nx)
        xt = yt[1:nx]; ut = reshape(yt[nx+1:end],nd,nu,nx)
        resu = computeresu(pde,coldata,t,x,xt,u,ut)
        resx = computeresx(pde,coldata,t,x,xt,u,ut)
        res[1:nx] = resx
        res[nx+1:end] = resu
        return res
    end

    y0 = vcat(x0,vec(u0))

    for (t,y,yt) in dasslIterator(dae,y0,pde.t0;args...)
        if t > 1
            return (t,y,yt)
        end
    end

end

end # module
