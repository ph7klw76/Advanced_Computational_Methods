import math
import numpy as np

g  = 9.81                  # m s⁻²
v0 = 70.0                  # m s⁻¹  (initial speed, magnitude)
m  = 0.50                  # kg
b  = 0.001                 # kg m⁻¹  (quadratic drag)

k  = b / m
X_TARGET = 200.0           # m

# -------------------------------------------------------------
# 1) Forward-Euler integrator (1st-order)
# -------------------------------------------------------------
def euler_step(x, y, vx, vy, dt):
    v  = math.hypot(vx, vy)
    ax = -k * v * vx
    ay = -g - k * v * vy
    return (
        x  + vx * dt,
        y  + vy * dt,
        vx + ax * dt,
        vy + ay * dt,
    )

# -------------------------------------------------------------
# 2) Horizontal range  R(θ)  (uses Euler)
# -------------------------------------------------------------
def range_for_angle(theta, dt=2e-4):
    x, y   = 0.0, 0.0
    vx, vy = v0 * math.cos(theta), v0 * math.sin(theta)

    while True:
        x_new, y_new, vx, vy = euler_step(x, y, vx, vy, dt)

        # detect first downward crossing of ground
        if y_new <= 0.0 and y > 0.0:
            frac = y / (y - y_new)      # linear interpolation
            return x + frac * (x_new - x)

        x, y = x_new, y_new

# -------------------------------------------------------------
# 3) Root function  f(θ) = R(θ) − X_TARGET
# -------------------------------------------------------------
def f(theta):
    return range_for_angle(theta) - X_TARGET

def fprime(theta, h=1e-4):
    return (range_for_angle(theta + h) - range_for_angle(theta - h)) / (2*h)

# -------------------------------------------------------------
# 4) Newton–Raphson solver for one bracket
# -------------------------------------------------------------
def newton(theta0, tol=1e-3, max_iter=25):
    theta = theta0
    for _ in range(max_iter):
        fn = f(theta)
        if abs(fn) < tol:
            return theta
        theta -= fn / fprime(theta)
    raise RuntimeError("Newton did not converge")

# -------------------------------------------------------------
# 5) Scan θ ∈ (0, 90°) to bracket every root, then refine
# -------------------------------------------------------------
grid = np.linspace(0.02, math.pi/2 - 0.02, 400)   # coarse grid
vals = [f(th) for th in grid]

roots = []
for th1, th2, v1, v2 in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
    if v1 * v2 < 0:                                # sign change → root inside
        theta_mid = 0.5 * (th1 + th2)
        roots.append(newton(theta_mid))

roots.sort()
print("\nLaunch angles (Euler dt = 2×10⁻⁴ s)\n")
for i, th in enumerate(roots, 1):
    print(f"θ{i}: {th: .8f} rad  =  {math.degrees(th): .5f}°   "
          f"(range = {range_for_angle(th):.3f} m)")

