T = 250
σ = 0.2
t = range(0, 20, length = T)
s = sin.(t)
y = s + randn(T) * σ

H = 1
A = 1

m = Model(y, H, A, 0.001, σ^2, 0.0, 1000.0)
kf = kalmanfilter(m)
ks = kalmansmoother(m, kf.priors, kf.posteriors)

using Plots

scatter(y, label = "Measurements", legend = :bottom,
        ylim = [-3, 2], markerstrokewidth = 0, markercolor = "black", foreground_color_legend = nothing)
plot!(s, label = "Noiseless signal", linewidth = 3, color = "grey")
plot!(kf.priors.μ, label = "Prediction", linewidth = 2, color = "blue")
plot!(kf.posteriors.μ, label = "Filter", linewidth = 2, color = "red")
plot!(ks.μ, label = "Smoother", linewidth = 2, color = "green")