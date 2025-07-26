import casadi as ca
from .DynamicsBase import DynamicsBase
# Example subclass
class CR3BPDynamics(DynamicsBase):
    def __init__(self, n_x=6, n_u=3):
        # Define parameter symbols specific to CR3BP
        self.mu    = ca.MX.sym('mu')
        super().__init__(n_x, n_u)

    def populate_state_dot(self):
        # Unpack state components
        r0, r1, r2 = self.states[0], self.states[1], self.states[2]
        v0, v1, v2 = self.states[3], self.states[4], self.states[5]
        # Unpack control components
        control0, control1, control2 = self.control[0], self.control[1], self.control[2]
        # Define distances to primaries
        len_r1 = ca.sqrt((r0 + self.mu)**2 + r1**2 + r2**2)
        len_r2 = ca.sqrt((r0 + self.mu - 1)**2 + r1**2 + r2**2)
        # Gravitational acceleration vector
        g0 = r0 - (1 - self.mu)*(r0 + self.mu)/len_r1**3 - self.mu*(r0 + self.mu - 1)/len_r2**3
        g1 = r1 - (1 - self.mu)*r1/len_r1**3 - self.mu*r1/len_r2**3
        g2 = - (1 - self.mu)*r2/len_r1**3 - self.mu*r2/len_r2**3
        g_vec = ca.vertcat(g0, g1, g2)
        # Coriolis terms
        h_vec = ca.vertcat(2*v1, -2*v0, 0)
        # Dynamics:
        r_dot = self.s*(ca.vertcat(v0, v1, v2))
        v_dot = self.s*(g_vec + h_vec) + ca.vertcat(control0, control1, control2)
        self.state_dot_expr = ca.vertcat(r_dot, v_dot)

    def configure_params(self, **param_values):
        """Configure parameters for CR3BP dynamics."""
        params = super().configure_params(**param_values)  # Get base class params
        if 'mu' in param_values:
            params['mu'] = param_values['mu']  # Use string key instead of CasADi symbol
        return params

    def _substitute_params(self, expr):
        # Use string key to avoid CasADi symbol boolean issues
        sub_expr = ca.substitute(expr, self.mu, self.params['mu'])
        sub_expr = super()._substitute_params(sub_expr)
        return sub_expr
