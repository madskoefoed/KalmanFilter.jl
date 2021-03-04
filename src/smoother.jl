function kalmansmoother(model::Model, priors::Output, posteriors::Output)
    T, m = size(posteriors.x)
    ks = Output(T, m)
    xt = priors.x[T + 1, :]
    Pt = priors.x[T + 1, :, :]
    for t in T:-1:1
        C             = posteriors.P[t, :, :] * model.A' * inv(priors.P[t + 1, :, :])
        ks.x[t, :]    = posteriors.x[t, :] + C * (xt - priors.x[t + 1, :])
        ks.P[t, :, :] = posteriors.P[t, :, :] + C * (Pt - priors.P[t + 1, :, :]) * C'
        xt = ks.x[t, :]
        Pt = ks.P[t, :, :]
    end
    return ks
end