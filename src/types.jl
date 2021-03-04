mutable struct Model{T<:Real}
    y::Vector{T}
    H::Vector{T}
    A::Matrix{T}
    Q::Matrix{T}
    R::T
    #function Model(y, H, A, Q, R)
        #R <= 0 && throw()
    #    new(y, H, A, Q, R)
    #end
end

mutable struct Priors{T<:Real}
    x::Vector{T}
    P::Matrix{T}
    #function Priors(x, P)
    #    !(length(x) == size(P, 1) == size(P, 2)) && throw(DimensionMismatch("Vector x is $(length(x))-dimensional and matrix P is a $(size(P)) matrix."))
    #    new(x, P)
    #end
end

mutable struct Output{T<:Real}
    x::Matrix{T}
    P::Array{T, 3}
end

Output(T::Int, m::Int) = Output(zeros(T, m), zeros(T, m, m))