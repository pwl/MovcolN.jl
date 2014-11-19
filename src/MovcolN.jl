module MovcolN

using Polynomials

export getcomputeres

include("polynomials.jl")       # standard polynomial related
                                # functions

# create matrices A and B such that A*Fi = B*Gi/Hi, assuming that Fi
# is evaluated at Gauss points and Gi is evaluated at Lobatto
# points.  The dimensions of A and B are n×n and n×n+1 respectively.
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


function computeux{T}(Qs,h::T,ul::Matrix{T},ur::Matrix{T})
    Qleft, Qright, _ = Qs
    nd, nu = size(ul)

    H  = [h^j for j = 0:nd-1]
    ux = Array(T,size(Qleft,1),nu)

    for i=1:nu
        ux[:,i] = (Qleft*(H.*ul[:,i]) + Qright*(H.*ur[:,i]))
        for j = 1:size(ux,1)
            # upscale spatial derivatives
            ux[j,i] /= h^(j-1)
        end
    end

    return ux
end


function computeutx(Qs,h,ht,xt,utl,utr,ux,ul,ur)
    Qleft, Qright, s = Qs
    nd, nu = size(utl)

    H  = [h^j for j = 0:nd-1]
    Ht = [j*ht*h^(j-1) for j = 0:nd-1]

    utx = zero(ux)

    for i=1:nu
        utx[:,i] = (Qleft*(H.*utl[:,i]+Ht.*ul[:,i]) + Qright*(H.*utr[:,i]+Ht.*ur[:,i]))
        xt = (xt+s*ht)
        for j = 1:size(utx,1)-1
            # upscale spatial derivatives
            utx[j,i] /= h^(j-1)
            # correct for the dilation and shift of the mesh
            utx[j,i] += -((j-1)*ht/h*ux[j,i]+xt*ux[j+1,i]) # j*ht*h^(j-1) = d/dt(h^j)
        end
    end

    return utx
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

function getcomputeres(F::Function,G::Function)

    T = Float64

    ns = 2                      # number of collocation points has to
                                # be even
    nu = 1                      # number of dependent variables
    nx = 2                      # number of mesh points
    kmax = ns+1                 # maximal darivative is of rank kmax-1
    sGauss, _ = gauss(ns)       # Gauss points
    sLobat, _ = lobatto(ns+1)   # Lobatto points

    QsGauss = map(x->generateQatx(ns,x,kmax=kmax),sGauss)
    QsLobat = map(x->generateQatx(ns,x,kmax=kmax),sLobat)

    FatGauss = Array(T,nu,ns,  nx)
    GatLobat = Array(T,nu,ns+1,nx)

    function computeres{T}(t :: T,
                           x :: Vector{T}, xt :: Vector{T},
                           u :: Array{T,3},ut :: Array{T,3})

        for i = 1:nx-1
            h  =  x[i+1]- x[i]
            ht = xt[i+1]-xt[i]

            for j = 1:ns
                ux  =  computeux(QsGauss[j],h,u[:,:,i],u[:,:,i+1])
                uxt = computeutx(QsGauss[j],h,ht,xt[i],ut[:,:,i],ut[:,:,i+1],ux,u[:,:,i],u[:,:,i+1])
                xj  = x[i]+sGauss[j]*h
                FatGauss[:,j,i] = F(t,xj,ux,uxt)
            end

            for j = 1:ns+1
                ux  =  computeux(QsLobat[j],h,u[:,:,i],u[:,:,i+1])
                uxt = computeutx(QsLobat[j],h,ht,xt[i],ut[:,:,i],ut[:,:,i+1],ux,u[:,:,i],u[:,:,i+1])
                xj  = x[i]+sLobat[j]*h
                GatLobat[:,j,i] = G(t,xj,ux,uxt)
            end

            @show A*F
            @show B*G/h

        end

    end
end

end # module
