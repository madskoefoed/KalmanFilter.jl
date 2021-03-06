function kalmansmoother(model::Model, priors::Output, posteriors::Output)
    T, m = size(posteriors.x)
    ks = Output(T, m)
    ks.x[T, :]    = posteriors.x[T, :]
    ks.P[T, :, :] = posteriors.P[T, :, :]
    for t in T-1:-1:1
        C             = posteriors.P[t, :, :] * model.A' * inv(priors.P[t + 1, :, :])
        ks.x[t, :]    = posteriors.x[t, :] + C * (ks.x[t, :] - priors.x[t + 1, :])
        ks.P[t, :, :] = posteriors.P[t, :, :] + C * (ks.P[t, :, :] - priors.P[t + 1, :, :]) * C'
    end
    return ks
end