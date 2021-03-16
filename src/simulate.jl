"""
```
Simulate!(Model)
```

Takes as input the type Model and simulates a state space model based on the specification in Model.

The function returns y and x containing the observation and state matrices.
"""

function simulate!(model::Model)
    T, p = size(model.y)
    m = length(model.x)

    x = zeros(T, m)
    y = zeros(T, p)

    xd = MvNormal(fill(0.0, m), model.Q)
    yd = MvNormal(fill(0.0, p), model.R)

    x[1, :] = model.A * model.x + rand(xd)
    y[1, :] = model.H * model.x + rand(yd)

    for t in 2:T
        x[t, :] = model.A * x[t - 1, :] + rand(xd)
        y[t, :] = model.H * x[t, :] + rand(yd)
    end
    return (y = y, x = x)
end