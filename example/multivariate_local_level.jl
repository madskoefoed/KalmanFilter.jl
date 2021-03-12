T = 250
Σ = [0.2 0.0; 0.0 0.8]
t = range(0, 20, length = T)
s = sin.(t)
y = s * [1 1] + randn(T, 2) * cholesky(Σ).L

H = Matrix(1.0I, 2, 2)
A = Matrix(1.0I, 2, 2)
Q = Matrix(0.01*I, 2, 2)
x = [0.0, 0.0]
P = Matrix(1000.0I, 2, 2)

m = Model(y, H, A, Q, Σ, x, P)
kf = kalmanfilter(m)
ks = kalmansmoother(m, kf.priors, kf.posteriors)

using Plots
l = @layout [a b]
p1 = scatter(t, y[:, 1], label = "Measurement", legend = :bottom, ylim = [-5, 4],
        title = "Variable 1", markerstrokewidth = 0, markercolor = "black", foreground_color_legend = nothing)
p1 = plot!(t, s, label = "Noiseless signal", linewidth = 3, color = "grey")
p1 = plot!(t, ks.μ[:, 1], label = "Smoother", linewidth = 2, color = :red)

p2 = scatter(t, y[:, 2], ylim = [-5, 4], legend = false, title = "Variable 2", markerstrokewidth = 0, markercolor = "black", foreground_color_legend = nothing)
p2 = plot!(t, s, linewidth = 3, color = "grey")
p2 = plot!(t, ks.μ[:, 2], linewidth = 2, color = :red)

plot(p1, p2, layout = l)