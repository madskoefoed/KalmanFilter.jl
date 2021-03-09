function kalmanfilter(model::Model)
    x = model.x
    P = model.P
    T, p = size(model.y)
    m = length(x)

    # Prepare priors and posteriors
    priors     = Output(T + 1, p, m)
    posteriors = Output(T, p, m)

    for t in 1:T
        # Prediction
        x, P, μ, Σ = time_update(x, P, model.H, model.A, model.Q, model.R)
        priors.x[t, :], priors.P[t, :, :], priors.μ[t, :], priors.Σ[t, :, :] = x, P, μ, Σ

        # Update
        x, P, μ, Σ = measurement_update(y[t, :], x, P, model.H, μ, Σ, model.R)
        posteriors.x[t, :], posteriors.P[t, :, :], posteriors.μ[t, :], posteriors.Σ[t, :, :] = x, P, μ, Σ

    end

    # Prediction at T + 1
    x, P, μ, Σ = time_update(x, P, model.H, model.A, model.Q, model.R)

    priors.x[T + 1, :], priors.P[T + 1, :, :], priors.μ[T + 1, :], priors.Σ[T + 1, :, :] = x, P, μ, Σ

    return (priors = priors, posteriors = posteriors)
end

predict_x(x, A) = A * x                   # Predicted state mean
predict_P(P, A, Q) = A * P * A' + Q       # Predicted state covariance
update_μ(x, H) = H * x                    # Measurement mean
update_Σ(P, H, R) = (H * P * H') + R      # Measurement variance 
update_K(P, H, Σ) = P * H' * inv(Σ)       # Kalman gain
update_x(x, y, μ, K) = x + K * (y - μ)  # Updated state mean
update_P(P, H, K) = (I - K * H) * P       # Updated state covariance

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