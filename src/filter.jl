function kalmanfilter(model::Model, priors::Priors)
    x = priors.x
    P = priors.P
    T = length(model.y)
    m = length(x)
    
    # Prepare priors and posteriors
    priors     = Output(T + 1, m)
    posteriors = Output(T, m)
    priors.x[1, :]    = x
    priors.P[1, :, :] = P

    for t in 1:T
        # Measurement update
        posteriors.x[t, :], posteriors.P[t, :, :] = measurement_update(model.y[t], priors.x[t, :], priors.P[t, :, :], model.H, model.R)
        # Time update
        priors.x[t + 1, :], priors.P[t + 1, :, :] = time_update(posteriors.x[t, :], posteriors.P[t, :, :], model.A, model.Q)
    end
    return (priors = priors, posteriors = posteriors)
end

function time_update(x, P, A, Q)
    x = A * x
    P = A * P * A' + Q
    return (x, P)
end

function measurement_update(y, x, P, H, R)
    # Kalman gain
    Q = (H' * P * H)[1] + R
    K = P * H / Q
    # Update estimate
    x = x .+ K .* (y - dot(H, x))
    # Update covariance
    P = (I - K' * H) * P
    return (x, P)
end