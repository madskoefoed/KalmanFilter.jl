# StateSpaceModels
# y(t)   = Z(t) * a(t) + e(t)        ~ N(0, H)
# a(t+1) = T(t) * a(t) + R(t) * n(t) ~ N(0, Q)

# Mine
# y(t)     = H * x(t) + e ~ N(0, R)
# x(t + 1) = A * x(t) + n ~ N(0, Q)

function kalmansmoother(model::Model, priors::Output, posteriors::Output)
    T, p = size(model.y)
    m = length(model.x)
    ks = Output(T, p, m)
    ks.x[T, :]    = posteriors.x[T, :]
    ks.P[T, :, :] = posteriors.P[T, :, :]
    ks.μ[T, :]    = update_μ(ks.x[T, :], model.H)
    ks.Σ[T, :, :] = update_Σ(ks.P[T, :, :], model.H, model.R)
    for t in T-1:-1:1
        J             = posteriors.P[t, :, :] * model.A' * inv(priors.P[t + 1, :, :])
        ks.x[t, :]    = posteriors.x[t, :] + J * (ks.x[t + 1, :] - priors.x[t + 1, :])
        ks.P[t, :, :] = posteriors.P[t, :, :] + J * (ks.P[t + 1, :, :] - priors.P[t + 1, :, :]) * J'
        ks.μ[t, :]    = update_μ(ks.x[t, :], model.H)
        ks.Σ[t, :, :] = update_Σ(ks.P[t, :, :], model.H, model.R)
    end
    return ks
end