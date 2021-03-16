using Plots
using Random

# Number of observations
T = 50

# Variances
Q = 0.1
R = 0.5

# Define model with prior N(0, 1000)
model = LocalLevel(zeros(T), Q, R)

# Simulate
model.y, x = simulate(model)

# Filter
kf = kalmanfilter(model)

# Smoother
ks = kalmansmoother(model, kf.priors, kf.posteriors)

# Plot
scatter(y, label = "Measurements", legend = :topright, title = "Univariate local level model",
        markerstrokewidth = 0, markercolor = "black", foreground_color_legend = nothing)
plot!(kf.posteriors.μ, label = "Filter", linewidth = 2, color = "red")
plot!(ks.μ, label = "Smoother", linewidth = 2, color = "green")