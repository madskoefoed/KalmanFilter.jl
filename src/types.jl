"""
```
Model(y, H, A, Q, R, x, P)
```

The type Model contains the measurements matrix as well as the kalman setup:
Measurement:  y(t)     = H * x(t) + e(t) ~ N(0, R)
State:        x(t + 1) = A * x(t) + n(t) ~ N(0, Q)

where both y(t) and x(t+1) are vectors of length p and m, respectively.
The state transition matrix, A, is (m x m) while the measurement model matrix, H, is (p x m).

Time-varying H and A are not currently implemented. Similarly, the covariances
R and Q are constant across time.
"""

mutable struct Model
    y::Matrix{<:Real}
    H::Matrix{<:Real}
    A::Matrix{<:Real}
    Q::Matrix{<:Real}
    R::Matrix{<:Real}
    x::Vector{<:Real}
    P::Matrix{<:Real}
    predicted::Predicted
    filtered::Filtered
    smoothed::Smoothed
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
        @assert all(diag(P) .> 0) "All diagonal elements of P must be strictly positive."

        predicted = Predicted(size(y, 1), size(y, 2), length(x))
        filtered  = Filtered( size(y, 1), size(y, 2), length(x))
        smoothed  = Smoothed( size(y, 1), size(y, 2), length(x))

        return new(y, H, A, Q, R, x, P, predicted, filtered, smoothed)
    end
end

# Outer constructors for univariate measurement
Model(y::Vector{<:Real}, H::Vector{<:Real}, A::Matrix{<:Real}, Q::Matrix{<:Real}, R::Real, x::Vector{<:Real}, P::Matrix{<:Real}) = Model(repeat(y, 1, 1), reshape(H, 1, length(H)), A, Q, fill(R, 1, 1), x, P)
Model(y::Vector{<:Real}, H::Real, A::Real, Q::Real, R::Real, x::Real, P::Real) = Model(repeat(y, 1, 1), fill(H, 1, 1), fill(A, 1, 1), fill(Q, 1, 1), fill(R, 1, 1), [x], fill(P, 1, 1))

# Univariate state
Model(y::Matrix{<:Real}, H::Matrix{<:Real}, A::Real, Q::Real, R::Matrix{<:Real}, x::Real, P::Real) = Model(y, H, fill(A, 1, 1), fill(Q, 1, 1), R, [x], fill(P, 1, 1))

# Special constructors
LocalLevel(y::Vector{<:Real}, Q::Real, R::Real, x::Real, P::Real) = Model(repeat(y, 1, 1), ones(1, 1), ones(1, 1), fill(Q, 1, 1), fill(R, 1, 1), [x], fill(P, 1, 1))
LocalLevel(y::Matrix{<:Real}, Q::Real, R::Matrix, x::Real, P::Real) = Model(y, ones(size(y, 2), 1), ones(1, 1), fill(Q, 1, 1), R, [x], fill(P, 1, 1))

LocalLevel(y::Vector{<:Real}, Q::Real, R::Real) = Model(repeat(y, 1, 1), ones(1, 1), ones(1, 1), fill(Q, 1, 1), fill(R, 1, 1), [y[1]], fill(1000.0, 1, 1))
LocalLevel(y::Matrix{<:Real}, Q::Real, R::Matrix) = Model(repeat(y, 1, 1), ones(1, 1), ones(1, 1), fill(Q, 1, 1), R, [0.0], fill(1000.0, 1, 1))

"""
```
Output
```

The type Output contains the means and covariances for the measurements
and states, respectively, at each time step, t = 1,...,T:
x is a (T x m) matrix of state means
P is a (T x m x m) array of state covariances
μ is a (T x p) matrix of measurement means
Σ is a (T x p x p) matrix of measurement covariances
"""

abstract type Output end

mutable struct Predicted <: Output
    x::Matrix{AbstractFloat}
    P::Array{AbstractFloat, 3}
    μ::Matrix{AbstractFloat}
    Σ::Array{AbstractFloat, 3}
end

mutable struct Filtered <: Output
    x::Matrix{AbstractFloat}
    P::Array{AbstractFloat, 3}
    μ::Matrix{AbstractFloat}
    Σ::Array{AbstractFloat, 3}
end

mutable struct Smoothed <: Output
    x::Matrix{AbstractFloat}
    P::Array{AbstractFloat, 3}
    μ::Matrix{AbstractFloat}
    Σ::Array{AbstractFloat, 3}
end

Predicted(T::Int, p::Int, m::Int) = Predicted(zeros(T, m), zeros(T, m, m), zeros(T, p), zeros(T, p, p))
Filtered(T::Int, p::Int, m::Int)  = Filtered(zeros(T, m), zeros(T, m, m), zeros(T, p), zeros(T, p, p))
Smoothed(T::Int, p::Int, m::Int)  = Smoothed(zeros(T, m), zeros(T, m, m), zeros(T, p), zeros(T, p, p))