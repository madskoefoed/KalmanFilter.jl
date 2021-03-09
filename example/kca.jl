T = 250
σ = 0.2
t = range(0, 20, length = T)
s = sin.(t)
y = s + randn(T) * σ

h = (t[end] - t[1])/T
H = [1.0, 0.0, 0.0]
A = [1.0 h   0.5*h^2;
     0.0 1.0 h;
     0.0 0.0 1.0]

m = Model(y, H, A, Matrix(0.01*I, 3, 3), σ^2, [0.0, 0.0, 0.0], Matrix(1000.0I, 3, 3))
kf = kalmanfilter(m)
ks = kalmansmoother(m, kf.priors, kf.posteriors)

using Plots

scatter(y, label = "Measurements", linewidth = 2, legend = :bottom, ylim = [-3, 2],
        markerstrokewidth = 0, markercolor = "black", foreground_color_legend = nothing)
plot!(s, label = "Noiseless signal", linewidth = 2, color = "grey")
plot!(kf.priors.μ, label = "Prediction", linewidth = 1, color = "blue")
plot!(kf.posteriors.μ, label = "Filter", linewidth = 1, color = "red")
plot!(ks.μ, label = "Smoother", linewidth = 1, color = "green")

scatter(y, label = "Measurements", linewidth = 2, legend = :bottom, ylim = [-3, 2],
        markerstrokewidth = 0, markercolor = "black", foreground_color_legend = nothing)
plot!(s, label = "Noiseless signal", linewidth = 2, color = "grey")
plot!(ks.x, label = ["Position" "Velocity" "Acceleration"], linewidth = 1, color = ["blue" "red" "green"])
