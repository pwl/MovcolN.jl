# generates the Lagrange basis for an array of roots at point x
function Lagrange(roots,x)
    nr    = length(roots)
    li    = Array(typeof(x),nr)

    for j = 1:nr
        li[j] = one(x)
        for i = 1:nr
            if i == j
                continue
            else
                li[j] *= (x-roots[i])/(roots[j]-roots[i])
            end
        end
    end

    return li
end


LegendreP(n::Int,x) = LegendrePgen(n::Int,x)[1]

function LegendrePgen{T}(n::Int,x::T)
    if n == 0
        return one(T), zero(T)
    else
        Pn1, Pn2 = LegendrePgen(n-1,x)
        return x*Pn1*(2n-1)/n-Pn2*(n-1)/n, Pn1
    end
end

# Generates hermite interpolation polynomials for the interval s∈[0,1]
# and valid for the first N derivatives (i.e. N=0 generates
# ps[1,1]=s-1 and ps[1,1]=s) etc.
function hermite01(n::Int)

    T = Rational{Int}
    e = eye(T,2n)
    M = zeros(T,2n,2n)

    for j = 1:2n
        p = Poly(e[:,j])
        for i1 = 0:n-1
            M[i1+1,  j] = polyval(polyder(p,i1),zero(T))
            M[i1+n+1,j] = polyval(polyder(p,i1), one(T))
        end
    end

    M⁻¹ = inv(M)                # here M⁻¹ is exact,
                                # typeof(M⁻¹)=Array{Rational{Int},2}

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
