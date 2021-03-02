function kalmansmoother(fo::FilterOutput)
    T  = length(fo.y)
    ks = SmootherOutput(fo.y, fo.x, fo.P, fo.H, fo.A, fo.Q, fo.R)
    A  = fo.A
    println(fo.x[1:5,:])
    for t in T:-1:1
        C             = fo.P[t, :, :] * A' * inv(fo.P[t + 1, :, :])
        ks.x[t, :]    = fo.x[t, :] + C * (ks.x[t + 1, :] - A * fo.x[t + 1, :])
        ks.P[t, :, :] = fo.P[t, :, :] + C * (ks.P[t + 1, :, :] - fo.P[t + 1, :, :]) * C'
    end
    println(ks.x[1:5,:])
    return ks
end
