using Distributions
using LinearAlgebra
using KalmanFilter
using Test

@testset "Example" begin
    
    T = 10_000
    σ = 0.5
    t = range(0, 20, length = T)
    h = (t[end] - t[1])/T
    H = [1.0, 0.0, 0.0]
    A = [1.0 h   0.5*h^2;
         0.0 1.0 h;
         0.0 0.0 1.0]
    s = sin.(t)
    y = s + randn(T) * σ
    
    m = Model(y, H, A, Matrix(0.1*I, 3, 3), σ^2)
    p = Priors([0.0, 0.0, 0.0], Matrix(1000.0I, 3, 3))
    kf = kalmanfilter(m, p)
    ks = kalmansmoother(m, kf.priors, kf.posteriors)

    @test isapprox(kf.posteriors.P[end, 1, 1], ks.P[end, 1, 1])
    @test isapprox(kf.posteriors.P[end, 2, 2], ks.P[end, 2, 2])
    @test isapprox(kf.posteriors.P[end, 3, 3], ks.P[end, 3, 3])

    RMSE = mean((y - kf.priors.x[1:end-1, :] * m.H).^2)

    #@test isapprox(RMSE, σ + )
end