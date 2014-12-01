module MovcolN

using Polynomials
using Devectorize
using DASSL
using ArrayViews
using Iterators

include("polynomials.jl")       # polynomial related functions

export Equation, movcol_solve

# Equation should consist of functions F,G,Bl,Br,M,g,tau,Bxl,Bxr,gamma
abstract Equation

immutable TemporaryArrays
    Hul :: Vector{Float64}
    Hur :: Vector{Float64}
    res :: Vector{Float64}
    ux  :: Matrix{Float64}
    utx :: Matrix{Float64}
end

function TemporaryArrays(nd,nu)
    TemporaryArrays(Array(Float64,nd),
                    Array(Float64,nd),
                    Array(Float64,nd+1),
                    Array(Float64,nd+1,nu),
                    Array(Float64,nd+1,nu))
end

immutable CollocationPoint
    left  :: Matrix{Float64}
    right :: Matrix{Float64}
    s     :: Float64
end

immutable MeshPoint
    x  :: Float64
    xt :: Float64
    u  :: AbstractMatrix{Float64}
    ut :: AbstractMatrix{Float64}
    nd :: Int
    nu :: Int
end

# CollocationData is the same for every mesh point and it depends on
# the number of collocation points (ns) only.
immutable CollocationData # only works with Float64
    ns :: Int        # number of derivatives provided at mesh points

    # vectors of tuples (Qleft, Qright, s)
    gauss   :: Vector{CollocationPoint}
    lobatto :: Vector{CollocationPoint}

    AB :: Matrix{Float64}
end


# @todo tests...
function CollocationData(ns)
    QsGauss   = map(s->generateQats(ns,s),  gauss(ns)[1])
    QsLobatto = map(s->generateQats(ns,s),lobatto(ns+1)[1])
    AB = generateAB(ns)
    return CollocationData(ns,QsGauss,QsLobatto,AB)
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
    return CollocationPoint(Qleft,Qright,s)
end


# given the vectors ul=[u,ux,uxx,...]_(x=x[N]) and
# ur=[u,ux,uxx,...]_(x=x[N+1]) generate a vector
# ux=[u,ux,uxx,...]_(x=x[N]+s*h), where s is prescribed in Qs (this
# descriptions of ul, ur and ux are oversimplification as each of them
# holds the derivatives of more than just one function,
# e.g. ul[j,i]=∂ⱼuᵢ(x[N]) ).  Generation of ux is realized by
# performing linear operations on ul and ur via the matrices Qleft and
# Qright.  The vector H is equal to [h^j for j=0:nd+1].
function computeux!{T}(ux::Array{T,2},Q::CollocationPoint,H::Vector{T},ul::AbstractMatrix{T},ur::AbstractMatrix{T},tmp)
    nd = size(ul,1)
    nu = size(ul,2)

    for i=1:nu
        @devec tmp.Hul[:] = H[1:nd].*ul[:,i]
        @devec tmp.Hur[:] = H[1:nd].*ur[:,i]
        mult_lr!(tmp.res,Q.left,Q.right,tmp.Hul,tmp.Hur)
        @devec ux[:,i] = tmp.res./H
    end
end

# Function similar to computeux! but computing
# utx=[ut,utx,utxx,...]. It requires more data (Ht,xt,utl,utr,ux)
# because it has to account for the mesh translation and rescaling in
# time.
function computeutx!{T}(utx::Array{T,2},Q::CollocationPoint,H::Vector{T},Ht::Vector{T},xt::T,utl,utr,ux,ul,ur,tmp)
    nd = size(utl,1)
    nu = size(utl,2)
    h  = H[2]
    ht = Ht[2]
    xt = (xt+Q.s*ht)

    for i = 1:nu
        @devec tmp.Hul[:] = H[1:nd].*utl[:,i]+Ht[1:nd].*ul[:,i]
        @devec tmp.Hur[:] = H[1:nd].*utr[:,i]+Ht[1:nd].*ur[:,i]
        mult_lr!(tmp.res,Q.left,Q.right,tmp.Hul,tmp.Hur)
        # upscale spatial derivatives
        scaling_utx!(utx,tmp.res,xt,ux,H,Ht,i)
    end
end

function mult_lr!(res,Qleft,Qright,Hul,Hur)
    nd = length(Hul)
    for i = 1:nd+1
        res[i] = 0
        for j = 1:nd
            @inbounds res[i] += Hul[j]*Qleft[i,j]+Hur[j]*Qright[i,j]
        end
    end
end

# correct for the mesh rescaling, dilation and shift of the mesh
function scaling_utx!(utx,res,xt,ux,H,Ht,i)
    for j = 1:length(res)-1
        @inbounds utx[j,i] = (res[j]-Ht[j]*ux[j,i])/H[j]-xt*ux[j+1,i]
    end
end

# For a given function F(t,u,ux,uxx,...,ut,utx,utxx,...) return the
# value F_(x=x[N]+s*h).  The spatial and temporal derivatives at
# x=x[N]+s*h are computed via computeux! and computeutx! functions.
function getcollocationvalues!{T}(FVal,Qs::Vector{CollocationPoint},F,t::T,left,right,H,Ht,tmp)
    h = H[2]

    for j = 1:size(FVal,1)
        s = Qs[j].s

        if s == zero(T)
            FVal[j,:] = F(t,left.x,left.u,left.ut)::Array{Float64,1}
        elseif s == zero(T)
            FVal[j,:] = F(t,right.x,right.u,right.ut)::Array{Float64,1}
        else
            x = left.x; xt = left.xt
            xs  = x+s*h
             computeux!(tmp.ux, Qs[j],H,left.u,right.u,tmp)
            computeutx!(tmp.utx,Qs[j],H,Ht,xt,left.ut,right.ut,tmp.ux,left.u,right.u,tmp)
            FVal[j,:] = F(t,xs,tmp.ux,tmp.utx)::Array{Float64,1}
        end

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
                        u  :: Array{T,3},ut :: Array{T,3};
                        args...)
    nd = size(u,1)
    nu = size(u,2)
    nx = size(u,3)
    ns = nd
    ns2 = div(ns,2)

    FGauss   = Array(T,ns  ,nu)
    GLobatto = Array(T,ns+1,nu)
    resu     = Array(T,ns  ,nu,nx)
    tmp      = TemporaryArrays(nd,nu)
    H        = Array(T,ns+1)
    Ht       = Array(T,ns+1)

    for i = 1:nx-1

        left  = MeshPoint(x[i  ],xt[i  ],view(u,:,:,i  ),view(ut,:,:,i  ), nd, nu)
        right = MeshPoint(x[i+1],xt[i+1],view(u,:,:,i+1),view(ut,:,:,i+1), nd, nu)
        h = x[i+1]-x[i];      ht = xt[i+1]-xt[i]

        for j = 0:nd
            H[j+1]  = h^j
            Ht[j+1] = j*ht*h^(j-1)
        end

        getcollocationvalues!(FGauss,  coldata.gauss,  pde.F,t,left,right,H,Ht,tmp)
        getcollocationvalues!(GLobatto,coldata.lobatto,pde.G,t,left,right,H,Ht,tmp)

        # equivalent to
        #
        # resu[:,:,i] = FGauss-coldata.AB*GLobatto/h
        #
        # for small AB the fillresu! is significantly faster than the
        # build-in matrix multiplication
        fillresu!(resu,FGauss,GLobatto,coldata.AB,h,i)

    end

    # residua for the boundary conditions, Bl and Br should each
    # return the vector of size at least nu*ns/2, the remaining
    # boundary conditions are discarded
    resBl = pde.Bl(t,x[1  ],xt[1  ],view(u,:,:,1 ),view(ut,:,:,1 ))::Array{Float64,1}
    resBr = pde.Br(t,x[end],xt[end],view(u,:,:,nx),view(ut,:,:,nx))::Array{Float64,1}
    resu[    1:ns2,:,nx] = reshape(resBl,nu,ns2)
    resu[ns2+1:ns, :,nx] = reshape(resBr,nu,ns2)

    return resu

end

function fillresu!(resu,F,G,AB,h,i)
    nG = size(G,1)
    nd = size(resu,1)
    nu = size(resu,2)
    for l = 1:nu
        for k = 1:nd
            resu[k,l,i] = F[k,l]
        end
    end
    for j = 1:nG
        for l = 1:nu
            for k = 1:nd
                resu[k,l,i] -= AB[k,j]*G[j,l]/h
            end
        end
    end
end


# compute residua for the mesh
function computeresx(pde :: Equation,
                     coldata :: CollocationData,
                     t :: Float64,
                     x :: Vector{Float64}, xt :: Vector{Float64},
                     u :: Array{Float64,3},ut :: Array{Float64,3};
                     args...)

    nd,nu,nx = size(u)
    ns = nd                     # number of collocation points
    ns2 = div(ns,2)

    uhalf    = Array(Float64,ns+1,nu)
    uthalf   = Array(Float64,ns+1,nu)
    resx     = Array(Float64,nx)
    Mhalf    = Array(Float64,nx-1)
    Yhalf    = Array(Float64,nx+1)
    tmp      = TemporaryArrays(nd,nu)

    Qshalf = coldata.lobatto[ns2+1]

    for i = 1:nx-1
        xl = x[i  ]; xtl = xt[i  ]; ul = view(u,:,:,i  ); utl = view(ut,:,:,i  )
        xr = x[i+1]; xtr = xt[i+1]; ur = view(u,:,:,i+1); utr = view(ut,:,:,i+1)
        h  = x[i+1]-x[i];      ht = xt[i+1]-xt[i]
        H  = [h^j for j=0:nd]
        Ht = [j*ht*h^(j-1) for j=0:nd]

        if i==1 || i==nx-1
            Mhalf[i] = ((pde.M(t,xl,ul)::Float64+pde.M(t,xr,ur)::Float64)/2)
        else
            xhalf  = (xl+xr)/2
            xthalf = (xtl+xtr)/2
             computeux!(uhalf, Qshalf,H,ul,ur,tmp)
            computeutx!(uthalf,Qshalf,H,Ht,xthalf,utl,utr,uhalf,ul,ur,tmp)
            Mhalf[i] = pde.M(t,xhalf,uhalf)::Float64
        end

        Yhalf[i+1] = -pde.tau*(xtr-xtl)/(xr-xl)^2+1/(xr-xl)

        if i == 1
            resx[ 1] = pde.Bxl(t,xl,xtl,ul,utl)::Float64
        end
        if i == nx-1
            resx[nx] = pde.Bxr(t,xr,xtr,ur,utr)::Float64
        end
    end

    # boundary conditions for Yhalf
    Yhalf[ 1 ] = Yhalf[  2  ]
    Yhalf[end] = Yhalf[end-1]

    Mhalf = smoothen(Mhalf;args...)

    # @todo add monitor smoothing

    # compute the residua for mesh movement
    for i = 2:nx-1
        g=pde.gamma*(pde.gamma+1)
        resx[i] = (Yhalf[i+1]-g*(Yhalf[i+2]-2*Yhalf[i+1]+Yhalf[i]))/Mhalf[i]-(Yhalf[i]-g*(Yhalf[i+1]-2*Yhalf[i]+Yhalf[i-1]))/Mhalf[i-1]
    end

    return resx

end


# simple gaussian filter to smoothen the monitor function
function smoothen(M; passes = 4, args...)
    if passes == 0
        return M
    else
        Msmooth = zero(M)
        @simd for i = 2:length(M)-1
            @inbounds Msmooth[i]=(M[i-1]+2M[i]+M[i+1])/4
        end
        Msmooth[1]=(M[1]+M[2])/2
        Msmooth[end]=(M[end-1]+M[end])/2
        return smoothen(Msmooth, passes=passes-1)
    end
end


# creates a relaxed mesh
function meshinit(pde :: Equation,
                  coldata :: CollocationData,
                  nx :: Int;
                  mesherr = 1e-5,
                  mesh_maxsteps = Inf,
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
        resx = computeresx(xpde,coldata,xpde.t0,x,xt,u,u;args...)
        return resx
    end

    nstp = 0
    for (t,x,xt) in dasslIterator(dae,x0,0.0;reltol=mesherr*1e-2,abstol=mesherr*1e-2)
        nstp += 1

        if norm(xt,Inf) < mesherr
            info("Mesh converged after $nstp steps")
            x0  = x
            break
        elseif nstp > mesh_maxsteps
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
        res = Array(Float64,nd*nu*nx+nx)
        # @todo xt = g(t)*xt, ut = g(t)*ut and make tau=tau(t)
         x =  y[1:nx];  u = reshape( y[nx+1:end],nd,nu,nx)
        xt = yt[1:nx]; ut = reshape(yt[nx+1:end],nd,nu,nx)
        resu = computeresu(pde,coldata,t,x,xt,u,ut;args...)
        resx = computeresx(pde,coldata,t,x,xt,u,ut;args...)
        res[1:nx] = resx
        res[nx+1:end] = resu
        return res
    end

    function get_results(t,y,yt)
         x =  y[1:nx];  u = reshape( y[nx+1:end],nd,nu,nx)
        xt = yt[1:nx]; ut = reshape(yt[nx+1:end],nd,nu,nx)
        return t, x, u, xt, ut
    end

    y0  = vcat(x0,vec(u0))
    sol = dasslIterator(dae,y0,pde.t0;args...)

    return imap(z->get_results(z...),sol)

end

end # module
