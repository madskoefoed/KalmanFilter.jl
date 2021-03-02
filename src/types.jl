mutable struct KalmanFilter{T<:Real}
    y::Vector{T}
    x::Vector{T}
    P::Matrix{T}
    H::Vector{T}
    A::Matrix{T}
    Q::Matrix{T}
    R::T
end

mutable struct FilterOutput{T<:Real}
    y::Vector{T}
    x::Matrix{T}
    P::Array{T, 3}
    H::Vector{T}
    A::Matrix{T}
    Q::Matrix{T}
    R::T
end

function FilterOutput(y, x, P, H, A, Q, R)
    m = length(kf.x)
    fo = FilterOutput(y, zeros(T + 1, m), zeros(T + 1, m, m), H, A, Q, R)
    fo.x[1, :] = x
    fo.P[1, :, :] = P
    return fo
end

mutable struct SmootherOutput{T<:Real}
    y::Vector{T}
    x::Matrix{T}
    P::Array{T, 3}
    H::Vector{T}
    A::Matrix{T}
    Q::Matrix{T}
    R::T
end

SmootherOutput(fo::FilterOutput) = SmootherOutput(y, zeros(T, length(fo.x)), zeros(T, length(fo.x), length(fo.x)), H, A, Q, R)