module KalmanFilter

using LinearAlgebra, Distributions

include("src/types.jl")
include("src/filter.jl")
include("src/smoother.jl")
include("example/example.jl")

# Export types
export Model, Priors, Output

# Export functions
export kalmanfilter, kalmansmooth

end # module
