
using Plots
H = [1.0]
A = Matrix(1.0I, 1, 1)
T = 100
y = zeros(T)
x = zeros(T, 1)
for t in 2:T
    x[t, :] = A * x[t - 1, :] + rand(MvNormal([0.0], [0.1]))
    y[t] = dot(H, x[t, :]) + rand(Normal(0.0, 0.2))
end

kf = KalmanFilter(y, [0.0], Matrix(1000.0I, 1, 1), H, A, Matrix(1.0I, 1, 1), 1.0)
fo = kalmanfilter(kf)
ks = kalmansmoother(fo)

scatter(y, ylim = [-2, 2], label = "Measurements") ; plot!(fo.x, label = "Filter") ; plot!(ks.x, label = "Smoother")

H = [1.0, 0.0]
A = [0.9 0; 0 0.8]
T = 100
y = zeros(T)
x = zeros(T, 2)
for t in 2:T
    x[t, :] = A * x[t - 1, :] + rand(MvNormal([0.0, 0.0], [0.1 0.0; 0.0 0.2]))
    y[t] = dot(H, x[t, :]) + rand(Normal(0.0, 1.0))
end
#plot(x) ; plot!(y)

#kf = KalmanFilter(y, [0., 0.], [100.0 0.0; 0.0 100.0], H, A, [1.0 0.0; 0.0 1.0], 1.0)
#fo = kalmanfilter(kf)
#kalmansmoother(fo)