mutable struct Model
    y::Matrix{<:Real}
    H::Matrix{<:Real}
    A::Matrix{<:Real}
    Q::Matrix{<:Real}
    R::Matrix{<:Real}
    x::Vector{<:Real}
    P::Matrix{<:Real}
    function Model(y, H, A, Q, R, x, P)
        str = "y is $(size(y, 1))x$(size(y, 2)), " *
              "H is $(size(H, 1))x$(size(H, 2)), " *
              "A is $(size(A, 1))x$(size(A, 2)), " *
              "Q is $(size(Q, 1))x$(size(Q, 2)), " *
              "R is $(size(R, 1))x$(size(R, 2)), " *
              "x is $(length(x)), " *
              "P is $(size(P, 1))x$(size(P, 2))"
        !(size(H, 2) == size(A, 1) == size(A, 2) == size(Q, 1) == size(Q, 2) == length(x) == size(P, 1) == size(P, 2)) && throw(DimensionMismatch(str))
        !(size(y, 2) == size(H, 1) == size(R, 1) == size(R, 2)) && throw(DimensionMismatch(str))

        @assert all(diag(R) .> 0) "All diagonal elements of R must be strictly positive."
        @assert all(diag(Q) .> 0) "All diagonal elements of Q must be strictly positive."
        return new(y, H, A, Q, R, x, P)
    end
end

# Outer constructors for univariate measurement
Model(y::Vector, H::Vector, A::Matrix, Q::Matrix, R::Real, x::Vector, P::Matrix) = Model(repeat(y, 1, 1), reshape(H, 1, length(H)), A, Q, fill(R, 1, 1), x, P)
Model(y::Vector, H::Real, A::Real, Q::Real, R::Real, x::Real, P::Real) = Model(repeat(y, 1, 1), fill(H, 1, 1), fill(A, 1, 1), fill(Q, 1, 1), fill(R, 1, 1), [x], fill(P, 1, 1))

# Outer constructors for univariate state
#Model(y::Matrix, H::Vector, A::Real, Q::Real, R::Matrix, x::Real, P::Real) = Model(y, reshape(H, 1, length(H)), fill(A, 1, 1), fill(Q, 1, 1), R, [x], fill(P, 1, 1))

# Special constructors
LocalLevel(y::Vector, Q::Real, R::Real, x::Real, P::Real) = Model(repeat(y, 1, 1), ones(1, 1), ones(1, 1), fill(Q, 1, 1), fill(R, 1, 1), [x], fill(P, 1, 1))
LocalLevel(y::Matrix, Q::Real, R::Matrix, x::Real, P::Real) = Model(y, ones(size(y, 2), 1), ones(1, 1), fill(Q, 1, 1), R, [x], fill(P, 1, 1))

mutable struct Output
    x::Matrix{AbstractFloat}
    P::Array{AbstractFloat, 3}
    μ::Matrix{AbstractFloat}
    Σ::Array{AbstractFloat, 3}
end

Output(T::Int, p::Int, m::Int) = Output(zeros(T, m), zeros(T, m, m), zeros(T, p), zeros(T, p, p))