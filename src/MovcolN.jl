module MovcolN

using Polynomials

# Generates hermite interpolation polynomials for the interval s∈[0,1]
# and valid for the first N derivatives (i.e. N=0 generates
# ps[1,1]=s-1 and ps[1,1]=s) etc.
function hermite01(n;T=Rational{Int})

    e  = eye(T,2n)
    M  = zeros(T,2n,2n)

    for j = 1:2n
        p = Poly(e[:,j])
        for i1 = 0:n-1
            M[i1+1,  j] = polyval(polyder(p,i1),zero(T))
            M[i1+n+1,j] = polyval(polyder(p,i1), one(T))
        end
    end

    M⁻¹ = inv(full(M))          # here M⁻¹ is exact, typeof(M⁻¹) =
                                # Array{Rational{Int},2}
    ps = [ Poly(M⁻¹*e[:,i]) for i = 1:2n]
    return reshape(ps,n,2)'

end

# generates Gauss points and weights on the interval [0,1], this
# function comes from the RosettaCode page
function gauss(n)
    λ, Q = eig(SymTridiagonal(zeros(n), [ i / sqrt(4i^2 - 1) for i = 1:n-1 ]))
    x = (λ + 1) * 1/2
    w = [ 2*Q[1,i]^2 for i = 1:n ]/2
    return x, w
end

# generates Lobatto points and weights on the interval s∈[0,1]
function lobatto(n)

    if n < 2
        error("Number of points should be at least 2.")
    end

    #1 Compute the exact Legendre polynomial with rational coefficients
    p = LegendreP(n-1, Poly(Rational{Int}[0,1]))
    dp = polyder(p,1)

    #2 get the collocation points by solving Pₙ₋₁'(xᵢ)=0.  The roots
    #function returns Float64.
    x = [-1, sort(roots(dp)), 1]

    #3 compute weights as 2/(n(n-1)(Pₙ₋₁(xᵢ))^2)
    px = polyval(p,x)
    w = 2./px.^2/(n*(n-1))

    # we need to rescale the results from the interval [-1,1] to [0,1]
    return (x.+1)/2, w/2

end

function LegendrePgen{T}(n::Int,x::T)
    if n == 0
        return one(T), zero(T)
    else
        Pn1, Pn2 = LegendrePgen(n-1,x)
        return x*Pn1*(2n-1)/n-Pn2*(n-1)/n, Pn1
    end
end

LegendreP(n::Int,x) = LegendrePgen(n::Int,x)[1]

# create matrices A and B such that A*Fi = B*Gi/Hi, assuming that Fi
# is evaluated at Gauss points and Gi is evaluated at Lobatto
# points.  The dimensions of A and B are n×n and n×n+1 respectively.
function generateAB(n)
    Li = [LegendreP(i,Poly(Rational{Int}[0,1])) for i = 0:n-1]
    x, _ = lobatto(n+1)
    # A = ∫Li dx with bounds [x[j],x[j+1]]
    A  = [polyval(polyint(Li[k]),x[j+1])-polyval(polyint(Li[k]),x[j]) for k=1:n, j=1:n]
    B  = spdiagm((fill(1,n),fill(-1,n)),(0,1))
    return A,B
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
    return Qleft, Qright
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


function computeutx{T}(Qs,h::T,ht::T,xt::T,utl,utr,ux,ul,ur)
    Qleft, Qright, s = Qs
    nd, nu = size(utl)

    H  = [h^j for j = 0:nd-1]
    Ht = [j*ht*h^(j-1) for j = 0:nd-1]
    @show Ht

    utx = zero(ux)

    for i=1:nu
        utx[:,i] = (Qleft*(H.*utl[:,i]+Ht.*ul[:,i]) + Qright*(H.*utr[:,i]+Ht.*ur[:,i]))
        @show utx
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

end # module
