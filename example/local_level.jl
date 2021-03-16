using Plots
using Random

# Define random seed
rng = MersenneTwister(123456)

# Number of observations
T = 100

# Variances
Q = 1.0
R = 0.5

# Distributions of state and observation
xd = Normal(0.0, sqrt(Q))
yd = Normal(0.0, sqrt(R))

# Initialization at time t = 1
x = zeros(T)
y = zeros(T)
for t = 2:T
        x[t] = x[t - 1] + rand(rng, xd)
        y[t] = x[t] + rand(rng, yd)
end

# Define model with prior N(0, 1000)
m = LocalLevel(y, Q, R)

# Filter
kf = kalmanfilter(m)

# Smoother
ks = kalmansmoother(m, kf.priors, kf.posteriors)

# Plot
scatter(y, label = "Measurements", legend = :topright, title = "Univariate local level model",
        markerstrokewidth = 0, markercolor = "black", foreground_color_legend = nothing)
plot!(kf.posteriors.μ, label = "Filter", linewidth = 2, color = "red")
plot!(ks.μ, label = "Smoother", linewidth = 2, color = "green")