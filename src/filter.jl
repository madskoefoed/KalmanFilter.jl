"""
```
Kalman Filter(Model)
```

Takes as input the type Model, which contains the measurement matrix
as well as the model specification and initial values (x0 and P0).

The function returns the types Prior and Posterior, containing the means
and covariances for the measurements and states for the prior and posterior,
respectively.
"""

function kalmanfilter(model::Model)
    x = model.x
    P = model.P
    T, p = size(model.y)
    m = length(x)

    # Store predicted and filtered values
    filt = Filtered(T, p, m)
    pred = Predicted(T, p, m)

    for t in 1:T
        # Prediction
        x, P, μ, Σ = time_update(x, P, model.H, model.A, model.Q, model.R)
        pred.x[t, :], pred.P[t, :, :], pred.μ[t, :], pred.Σ[t, :, :] = x, P, μ, Σ

        # Update
        x, P, μ, Σ = measurement_update(model.y[t, :], x, P, model.H, μ, Σ, model.R)
        filt.x[t, :], filt.P[t, :, :], filt.μ[t, :], filt.Σ[t, :, :] = x, P, μ, Σ
    end
    return (predicted = pred, filtered = filt)
end

predict_x(x, A) = A * x                # Predicted state mean
predict_P(P, A, Q) = A * P * A' + Q    # Predicted state covariance
update_μ(x, H) = H * x                 # Measurement mean
update_Σ(P, H, R) = (H * P * H') + R   # Measurement variance 
update_K(P, H, Σ) = P * H' * inv(Σ)    # Kalman gain
update_x(x, y, μ, K) = x + K * (y - μ) # Updated state mean
update_P(P, H, K) = (I - K * H) * P    # Updated state covariance

function time_update(x, P, H, A, Q, R)
    x = predict_x(x, A)
    P = predict_P(P, A, Q)
    μ = update_μ(x, H)
    Σ = update_Σ(P, H, R)
    return (x, P, μ, Σ)
end

function measurement_update(y, x, P, H, μ, Σ, R)
    K = update_K(P, H, Σ)
    x = update_x(x, y, μ, K)
    P = update_P(P, H, K)
    μ = update_μ(x, H)
    Σ = update_Σ(P, H, R)
    return (x, P, μ, Σ)
end