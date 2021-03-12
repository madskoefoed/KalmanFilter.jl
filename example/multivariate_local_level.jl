T = 250
Σ = [0.2 0.0; 0.0 0.4]
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
r1 = (ks.μ[:, 1] .- 2*sqrt.(ks.Σ[:, 1, 1]), ks.μ[:, 1] .+ 2*sqrt.(ks.Σ[:, 1, 1]))
r2 = (ks.μ[:, 2] .- 2*sqrt.(ks.Σ[:, 2, 2]), ks.μ[:, 2] .+ 2*sqrt.(ks.Σ[:, 2, 2]))

using Plots
l = @layout [a b]
p1 = scatter(t, y[:, 1], label = "Measurement(1)", legend = :bottom, ylim = [-3, 2],
        markerstrokewidth = 0, markercolor = "black", foreground_color_legend = nothing, ribbon = r1)
p1 = plot!(t, s, label = "Noiseless signal", linewidth = 2, color = "grey")
p1 = plot!(t, ks.μ[:, 1], label = "Smoother", linewidth = 2, color = :blue)

p2 = scatter(t, y[:, 2], label = "Measurement(2)", legend = :bottom, ylim = [-3, 2],
        markerstrokewidth = 0, markercolor = "black", foreground_color_legend = nothing, ribbon = r2)
p2 = plot!(t, s, label = "Noiseless signal", linewidth = 2, color = "grey")
p2 = plot!(t, ks.μ[:, 2], label = "Smoother", linewidth = 2, color = :blue)

plot(p1, p2, layout = l)
