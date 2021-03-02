module KalmanFilter

using LinearAlgebra, Distributions

include("src/types.jl")
include("src/filter.jl")
include("src/smoother.jl")
include("example/local_level.jl")

# Export types
export KalmanFilter

# Export functions
export filter, smooth

end # module
