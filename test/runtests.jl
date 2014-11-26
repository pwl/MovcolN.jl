module MovcolNTests

using MovcolN
using FactCheck

# write your own tests here

facts("Generating Q") do
    kmax  = 4; nder = 3;
    Ql,Qr,x = MovcolN.generateQats(nder,1//2,kmax=kmax)

    @fact x => 1//2

    @fact size(Ql) => (kmax,nder)

    @fact Ql => [[1//2,5//32,1//64] [-15//8,-7//16,-1//32] [0//1,-3//2,-1//4] [30//1,15//1,3//2]]'

    @fact Qr => [[1//2,-5//32,1//64] [15//8,-7//16,1//32] [0//1,3//2,-1//4] [-30//1,15//1,-3//2]]'


end

facts("Computing derivatives") do
    kmax    = 4; nder = 3;
    Ql,Qr,_ = MovcolN.generateQats(nder,0.5,kmax=kmax)

    ul = hcat(Float64[ 0, 0, 2   ])
    ur = hcat(Float64[ 4, 4, 2   ])
    utl= hcat(Float64[ 0, 0, 2   ])
    utr= hcat(Float64[12, 8, 2   ])
    uc = hcat(Float64[ 1, 2, 2, 0])
    utc=      Float64[ 1, 2, 2   ]
    h  = 2.; ht = 2.
    H  = [h^j for j = 0:kmax]
    Ht = [j*ht*h^(j-1) for j = 0:kmax]

    uc1 = Array(Float64,kmax,1)
    utc1 = Array(Float64,kmax,1)

    MovcolN.computeux!(uc1,(Ql,Qr,0.5),H,ul,ur)
    MovcolN.computeutx!(utc1,(Ql,Qr,0.5),H,Ht,0.,utl,utr,uc,ul,ur)

    @fact uc1  => uc
    @fact utc1[1:3] => utc
end

facts("Generating AB matrix") do
    relerr = Array(Float64,0)
    abserr = Array(Float64,0)

    # for Float64 this method converges for up to 8 collocatin points
    nconv = 8
    for n = 2:nconv
        AB  =MovcolN.generateAB(n)
        sl,_=MovcolN.lobatto(n+1)
        sg,_=MovcolN.gauss(n)

        @fact size(AB) => (n,n+1)

        push!(abserr,norm((AB*sin(sl)-cos(sg)),         Inf))
        push!(relerr,norm((AB*sin(sl)-cos(sg))./sin(sg),Inf))
    end

    relconv = exp(diff(log(relerr)))
    absconv = exp(diff(log(abserr)))

    @fact relconv.<0.2 => all  # relative error is at most ~O(0.2^n)
    @fact absconv.<0.1 => all  # absolute error is at most ~O(0.1^n)
end

facts("MeshPair sanity check") do

    nx = 2
    nd = 4
    u0(x) = [x^3,3x^2,6x,6]
    ut0(x) = [6x,6,0,0]
    x  = Float64[0,1]
    xt = zero(x)
    u  = reshape(vcat(map(u0, x)...),nd,1,nx)
    ut = reshape(vcat(map(ut0,x)...),nd,1,nx)
    y  = [x,vec(u)]
    yt = [xt,vec(ut)]

    mp=MovcolN.MeshPair(x,xt,u,ut,1)

    @fact mp.nd => nd
    @fact mp.nu => 1
    @fact mp.h  => 1
    @fact mp.ht => 0
    @fact mp.H  => Float64[1,1,1,1,1]
    @fact mp.Ht => Float64[0,0,0,0,0]
    @fact mp.left  =>(0.,0.,hcat(u0(0.)),hcat(ut0(0.)))
    @fact mp.right =>(1.,0.,hcat(u0(1.)),hcat(ut0(1.)))

end

end
