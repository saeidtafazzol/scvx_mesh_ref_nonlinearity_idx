import numpy as np
import casadi as ca
from .VanillaDynamicsBase import VanillaDynamicsBase

class VanillaCR3BPDynamics(VanillaDynamicsBase):
    """
    Vanilla CR3BP dynamics without convex optimization features.
    Implementation matches the original CR3BPDynamics exactly.
    """
    def __init__(self, n_x=6, n_u=3):
        # Define parameter symbols
        self.mu = ca.MX.sym('mu')
        self.c = ca.MX.sym('c')
        super().__init__(n_x, n_u)
        
    def populate_expressions(self):
        """
        Populate symbolic expressions for CR3BP dynamics.
        Exact match to original implementation.
        """
        # Unpack state components (matching original variable names)
        r0, r1, r2 = self.states[0], self.states[1], self.states[2]
        v0, v1, v2 = self.states[3], self.states[4], self.states[5]
        
        # Unpack control components (matching original)
        control0, control1, control2 = self.control[0], self.control[1], self.control[2]
        
        # Define distances to primaries (exact match to original)
        len_r1 = ca.sqrt((r0 + self.mu)**2 + r1**2 + r2**2)
        len_r2 = ca.sqrt((r0 + self.mu - 1)**2 + r1**2 + r2**2)
        
        # Gravitational acceleration vector (exact match to original)
        g0 = r0 - (1 - self.mu) * (r0 + self.mu) / len_r1**3 - self.mu * (r0 + self.mu - 1) / len_r2**3
        g1 = r1 - (1 - self.mu) * r1 / len_r1**3 - self.mu * r1 / len_r2**3
        g2 = -(1 - self.mu) * r2 / len_r1**3 - self.mu * r2 / len_r2**3
        g_vec = ca.vertcat(g0, g1, g2)
        
        # Coriolis terms (exact match to original)
        h_vec = ca.vertcat(2 * v1, -2 * v0, 0)
        
        # Dynamics (with proper thrust acceleration = force/mass)
        r_dot = ca.vertcat(v0, v1, v2)
        # Control represents thrust force, so acceleration = thrust/mass
        thrust_acceleration = ca.vertcat(control0, control1, control2) / self.mass
        v_dot = g_vec + h_vec + thrust_acceleration
        
        self.state_dot_expr = ca.vertcat(r_dot, v_dot)
        
        # Mass dynamics
        control_len = ca.sqrt(control0**2 + control1**2 + control2**2)
        self.m_dot_expr = -control_len / self.c
        
    def _substitute_params(self, expr):
        """Substitute parameter values into expression."""
        if 'mu' in self.params:
            expr = ca.substitute(expr, self.mu, self.params['mu'])
        if 'c' in self.params:
            expr = ca.substitute(expr, self.c, self.params['c'])
        return expr

