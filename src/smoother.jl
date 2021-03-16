"""
```
Kalman Smoother(Model, Output, Output)
```

Takes as input the output from the Kalman Filter, i.e. the predicted
and filtered of the means and covariances of the states and measurements.

The function returns the type Output which smoothed means and covariances
for the states and measurements, respectively.
"""

function kalmansmoother(model::Model, predicted::Predicted, filtered::Filtered)
    T, p = size(model.y)
    m = length(model.x)
    ks = Smoother(T, p, m)
    ks.x[T, :]    = filtered.x[T, :]
    ks.P[T, :, :] = filtered.P[T, :, :]
    ks.μ[T, :]    = update_μ(ks.x[T, :], model.H)
    ks.Σ[T, :, :] = update_Σ(ks.P[T, :, :], model.H, model.R)
    for t in T-1:-1:1
        J             = filtered.P[t, :, :] * model.A' * inv(predicted.P[t + 1, :, :])
        ks.x[t, :]    = filtered.x[t, :] + J * (ks.x[t + 1, :] - predicted.x[t + 1, :])
        ks.P[t, :, :] = filtered.P[t, :, :] + J * (ks.P[t + 1, :, :] - predicted.P[t + 1, :, :]) * J'
        ks.μ[t, :]    = update_μ(ks.x[t, :], model.H)
        ks.Σ[t, :, :] = update_Σ(ks.P[t, :, :], model.H, model.R)
    end
    return ks
end