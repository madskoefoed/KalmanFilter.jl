using Plots
T = 100
σ = 0.2
t = range(0, 20, length = T)
h = (t[end] - t[1])/T
H = [1.0, 0.0, 0.0]
A = [1.0 h   0.5*h^2;
     0.0 1.0 h;
     0.0 0.0 1.0]
s = sin.(t)
y = s + randn(T) * σ

m = Model(y, H, A, Matrix(0.01*I, 3, 3), σ^2)
p = Priors([0.0, 0.0, 0.0], Matrix(1000.0I, 3, 3))
kf = kalmanfilter(m, p)
ks = kalmansmoother(m, kf.priors, kf.posteriors)

scatter(y, label = "Measurements", linewidth = 2, legend = :bottomleft, ylim = [-2.5, 1.5], markerstrokewidth = 0, markercolor = "black")
plot!(s, label = "Noiseless signal", linewidth = 2, color = "grey")
plot!(kf.priors.x * H, label = "Prediction", linewidth = 1, color = "blue")
plot!(kf.posteriors.x * H, label = "Filter", linewidth = 1, color = "red")
plot!(ks.x * H, label = "Smoother", linewidth = 1, color = "green")