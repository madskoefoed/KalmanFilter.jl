using Distributions
using LinearAlgebra
using KalmanFilter
using Test

@testset "Dimensions check" begin
    M = 1:3
    P = 1:3
    for p in P
        for m in M
            y = rand(10, p)
            H = rand(p, m)
            A = rand(m, m)
            Q = Matrix(Diagonal(fill(1.0, m)))
            R = Matrix(Diagonal(fill(1.0, p)))
            x = rand(m)
            P = Matrix(Diagonal(fill(1.0, m)))
            model = Model(y, H, A, Q, R, x, P)

            # Test p dimension
            @test p == size(model.H, 1) == size(model.R, 1) == size(model.R, 2)
            # Test m dimension
            @test m == size(model.H, 2) == size(model.Q, 1) == size(model.Q, 2) == length(model.x) == size(model.P, 1) == size(model.P, 2) == size(model.A,1) == size(model.A, 2)
        end
    end
end

@testset "Univairate Local Level" begin
    T = 100
    Q = 0.1
    R = 4.0
    x = 0.0
    P = 1000.0
    y = rand(Normal(0.0, sqrt(R)), T)

    model = LocalLevel(y, Q, R, x, P)
    kf = kalmanfilter(model)
    ks = kalmansmoother(model, kf.priors, kf.posteriors)

    @test isapprox(kf.posteriors.x[end, 1], ks.x[end, 1])
    @test isapprox(kf.posteriors.P[end, 1, 1], ks.P[end, 1, 1])

    @test isapprox(model.H[1, 1], 1.0)
    @test isapprox(model.A[1, 1], 1.0)
end