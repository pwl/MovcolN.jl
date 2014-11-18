using MovcolN
using FactCheck

import MovcolN

# write your own tests here

facts("Generating Q") do
    kmax  = 4; nder = 3;
    Ql,Qr = MovcolN.generateQatx(nder,1//2,kmax=kmax)

    @fact size(Ql) => (kmax,nder)

    @fact Ql => [[1//2,5//32,1//64] [-15//8,-7//16,-1//32] [0//1,-3//2,-1//4] [30//1,15//1,3//2]]'

    @fact Qr => [[1//2,-5//32,1//64] [15//8,-7//16,1//32] [0//1,3//2,-1//4] [-30//1,15//1,-3//2]]'


end

facts("Computing derivatives") do
    kmax  = 4; nder = 3;
    Ql,Qr = MovcolN.generateQatx(nder,0.5,kmax=kmax)

    ul = hcat(Float64[ 0, 0, 2   ])
    ur = hcat(Float64[ 4, 4, 2   ])
    utl= hcat(Float64[ 0, 0, 2   ])
    utr= hcat(Float64[12, 8, 2   ])
    uc = hcat(Float64[ 1, 2, 2, 0])
    utc=      Float64[ 1, 2, 2   ]

    uc1  = MovcolN.computeux((Ql,Qr,0.5),2.,ul,ur)
    utc1 = MovcolN.computeutx((Ql,Qr,0.5),2.,2.,0.,utl,utr,uc,ul,ur)

    @fact uc1  => uc
    @fact utc1[1:3] => utc
end
