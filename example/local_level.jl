
using Plots
H = [1.0]
A = Matrix(1.0I, 1, 1)
T = 25
y = zeros(T);
x = zeros(T, 1);
for t in 2:T
    x[t, :] = A * x[t - 1, :] + rand(MvNormal([0.0], [0.1]))
    y[t] = dot(H, x[t, :]) + rand(Normal(0.0, 0.2))
end

m = Model(y, H, A, Matrix(1.0I, 1, 1), 1.0)
p = Priors([0.0], Matrix(1000.0I, 1, 1))
kf = kalmanfilter(m, p)
ks = kalmansmoother(m, kf.priors, kf.posteriors)

scatter(y, ylim = [-2, 2], label = "Measurements", linewidth = 2, legend = :bottom)
plot!(kf.priors.x, label = "Prediction", linewidth = 2)
plot!(ss.filter.a, linewidth = 2, label = "Package")

scatter(y, ylim = [-2, 2], label = "Measurements", linewidth = 2, legend = :bottom)
plot!(kf.posteriors.x, label = "Filter", linewidth = 2)
plot!(ss.filter.att, linewidth = 2, label = "Package")

plot!(kf.posteriors.x, label = "Filter", linewidth = 2)
plot!(ks.x, label = "Smoother", linewidth = 2)