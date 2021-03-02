function kalmanfilter(kf::KalmanFilter)
    T  = length(kf.y)
    fo = FilterOutput(kf.y, kf.x, kf.P, kf.H, kf.A, kf.Q, kf.R)
    for t in 1:T
        # Time update
        fo.x[t + 1, :], fo.P[t + 1, :, :] = time_update(fo.x[t, :], fo.P[t, :, :], fo.A, fo.Q)
        # Measurement update
        fo.x[t + 1, :], fo.P[t + 1, :, :] = measurement_update(fo.y[t], fo.x[t + 1, :], fo.P[t + 1, :, :], fo.H, fo.R)
    end
    return fo
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