import casadi as ca
from .DynamicsBase import DynamicsBase


class TwoBodyMEE(DynamicsBase):
    """Two-body dynamics in Modified Equinoctial Elements (MEE)"""
    def __init__(self, n_x=6, n_u=3):
        super().__init__(n_x, n_u)

    def populate_state_dot(self):
        p, f, g, h, k, L = self.states[0], self.states[1], self.states[2], self.states[3], self.states[4], self.states[5]
        control0, control1, control2 = self.control[0], self.control[1], self.control[2]
        q = 1 + f * ca.cos(L) + g * ca.sin(L)
        s2 = 1 + h**2 + k**2
        A = ca.vertcat(0, 0, 0, 0, 0, ca.sqrt(p) * (q / p)**2)
        B = ca.vertcat(
            ca.horzcat(0,               2*p/q * ca.sqrt(p),                              0),
            ca.horzcat(ca.sqrt(p)*ca.sin(L), ca.sqrt(p)/q * ((q+1)*ca.cos(L)+f), -ca.sqrt(p)*g/q * (h*ca.sin(L)-k*ca.cos(L))),
            ca.horzcat(-ca.sqrt(p)*ca.cos(L), ca.sqrt(p)/q * ((q+1)*ca.sin(L)+g), ca.sqrt(p)*f/q * (h*ca.sin(L)-k*ca.cos(L))),
            ca.horzcat(0, 0, ca.sqrt(p)*s2*ca.cos(L)/2/q),
            ca.horzcat(0, 0, ca.sqrt(p)*s2*ca.sin(L)/2/q),
            ca.horzcat(0, 0, ca.sqrt(p)/q*(h*ca.sin(L)-k*ca.cos(L)))
        )
        self.state_dot_expr = ca.mtimes(B, ca.vertcat(control0, control1, control2)) + self.s * A

    def configure_params(self, **param_values):
        """Configure parameters for TwoBody dynamics."""
        return super().configure_params(**param_values)  # Uses base class params only

    def _substitute_params(self, expr):
        return super()._substitute_params(expr)


class TwoBodyCartesian(DynamicsBase):
    """Two-body dynamics in Cartesian coordinates"""
    def __init__(self, n_x=6, n_u=3):
        # Define parameter symbols specific to two-body dynamics
        self.mu = ca.MX.sym('mu')  # Gravitational parameter
        super().__init__(n_x, n_u)

    def populate_state_dot(self):
        # Unpack state components (position and velocity)
        r0, r1, r2 = self.states[0], self.states[1], self.states[2]
        v0, v1, v2 = self.states[3], self.states[4], self.states[5]
        # Unpack control components
        control0, control1, control2 = self.control[0], self.control[1], self.control[2]
        
        # Distance from central body
        r_mag = ca.sqrt(r0**2 + r1**2 + r2**2)
        
        # Gravitational acceleration vector
        g_vec = -self.mu / r_mag**3 * ca.vertcat(r0, r1, r2)
        
        # Dynamics:
        r_dot = self.s * ca.vertcat(v0, v1, v2)
        v_dot = self.s * g_vec + ca.vertcat(control0, control1, control2)
        
        self.state_dot_expr = ca.vertcat(r_dot, v_dot)

    def configure_params(self, **param_values):
        """Configure parameters for Cartesian two-body dynamics."""
        params = super().configure_params(**param_values)  # Get base class params
        if 'mu' in param_values:
            params[self.mu] = param_values['mu']
        return params

    def _substitute_params(self, expr):
        sub_expr = ca.substitute(expr, self.mu, self.params[self.mu])
        sub_expr = super()._substitute_params(sub_expr)
        return sub_expr

