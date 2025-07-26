# Vanilla Dynamics

Clean dynamics implementations without convex optimization features for sanity checking optimization results.

## Purpose

Validate trajectory optimization by propagating with the actual, unmodified dynamics. No time dilation, derivatives, or z-dot formulations.

## Files

- `VanillaDynamicsBase.py` - Base class with compilation for `state_dot()` and `m_dot()`
- `VanillaCR3BPDynamics.py` - CR3BP dynamics (exact match to original implementation)  
- `VanillaTwoBodyDynamics.py` - Two-body dynamics in MEE and Cartesian (exact match to originals)

## Usage

```python
# Initialize dynamics
dynamics = VanillaCR3BPDynamics()
dynamics.set_params(mu=0.012277471, c=3000.0)
dynamics.compile()  # Compile to shared libraries

# Use compiled functions
state_derivative = dynamics.state_dot(state, control, mass)
mass_derivative = dynamics.m_dot(control)
```

## Parameters

- **CR3BP**: `mu` (mass ratio), `c` (characteristic velocity)
- **Two-Body**: `mu` (gravitational parameter), `c` (characteristic velocity)
