import numpy as np
import casadi as ca
from .VanillaDynamicsBase import VanillaDynamicsBase

class VanillaTwoBodyMEE(VanillaDynamicsBase):
    """
    Vanilla Two-Body dynamics in Modified Equinoctial Elements (MEE).
    Implementation matches the original TwoBodyMEE exactly.
    """
    def __init__(self, n_x=6, n_u=3):
        # Define parameter symbols
        self.c = ca.MX.sym('c')
        super().__init__(n_x, n_u)
        
    def populate_expressions(self):
        """
        Populate symbolic expressions for MEE dynamics.
        Exact match to original implementation.
        """
        # Unpack state (exact match to original variable names)
        p, f, g, h, k, L = self.states[0], self.states[1], self.states[2], self.states[3], self.states[4], self.states[5]
        
        # Unpack control (exact match to original)
        control0, control1, control2 = self.control[0], self.control[1], self.control[2]
        
        # Intermediate calculations (exact match to original)
        q = 1 + f * ca.cos(L) + g * ca.sin(L)
        s2 = 1 + h**2 + k**2
        
        # Unperturbed motion vector A (exact match to original)
        A = ca.vertcat(0, 0, 0, 0, 0, ca.sqrt(p) * (q / p)**2)
        
        # Perturbation matrix B (exact match to original)
        B = ca.vertcat(
            ca.horzcat(0, 2*p/q * ca.sqrt(p), 0),
            ca.horzcat(ca.sqrt(p)*ca.sin(L), ca.sqrt(p)/q * ((q+1)*ca.cos(L)+f), -ca.sqrt(p)*g/q * (h*ca.sin(L)-k*ca.cos(L))),
            ca.horzcat(-ca.sqrt(p)*ca.cos(L), ca.sqrt(p)/q * ((q+1)*ca.sin(L)+g), ca.sqrt(p)*f/q * (h*ca.sin(L)-k*ca.cos(L))),
            ca.horzcat(0, 0, ca.sqrt(p)*s2*ca.cos(L)/2/q),
            ca.horzcat(0, 0, ca.sqrt(p)*s2*ca.sin(L)/2/q),
            ca.horzcat(0, 0, ca.sqrt(p)/q*(h*ca.sin(L)-k*ca.cos(L)))
        )
        
        # Dynamics (with proper thrust acceleration = force/mass)
        # Control represents thrust force, so we need to divide by mass
        thrust_control = ca.vertcat(control0, control1, control2) / self.mass
        self.state_dot_expr = ca.mtimes(B, thrust_control) + A
        
        # Mass dynamics
        control_len = ca.sqrt(control0**2 + control1**2 + control2**2)
        self.m_dot_expr = -control_len / self.c
        
    def _substitute_params(self, expr):
        """Substitute parameter values into expression."""
        if 'c' in self.params:
            expr = ca.substitute(expr, self.c, self.params['c'])
        return expr


class VanillaTwoBodyCartesian(VanillaDynamicsBase):
    """
    Vanilla Two-Body dynamics in Cartesian coordinates.
    Implementation matches the original TwoBodyCartesian exactly.
    """
    def __init__(self, n_x=6, n_u=3):
        # Define parameter symbols
        self.mu = ca.MX.sym('mu')
        self.c = ca.MX.sym('c')
        super().__init__(n_x, n_u)
        
    def populate_expressions(self):
        """
        Populate symbolic expressions for Cartesian two-body dynamics.
        Exact match to original implementation.
        """
        # Unpack state components (exact match to original variable names)
        r0, r1, r2 = self.states[0], self.states[1], self.states[2]
        v0, v1, v2 = self.states[3], self.states[4], self.states[5]
        
        # Unpack control components (exact match to original)
        control0, control1, control2 = self.control[0], self.control[1], self.control[2]
        
        # Distance from central body (exact match to original)
        r_mag = ca.sqrt(r0**2 + r1**2 + r2**2)
        
        # Gravitational acceleration vector (exact match to original)
        g_vec = -self.mu / r_mag**3 * ca.vertcat(r0, r1, r2)
        
        # Dynamics (with proper thrust acceleration = force/mass)
        r_dot = ca.vertcat(v0, v1, v2)
        # Control represents thrust force, so acceleration = thrust/mass
        thrust_acceleration = ca.vertcat(control0, control1, control2) / self.mass
        v_dot = g_vec + thrust_acceleration
        
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
