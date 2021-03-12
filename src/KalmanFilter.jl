module KalmanFilter

using LinearAlgebra, Distributions

include("src/types.jl")
include("src/filter.jl")
include("src/smoother.jl")
include("example/local_level.jl")
include("example/kca.jl")

# Export types
export Model, Output

# Export functions
export kalmanfilter, kalmansmooth

end # module
