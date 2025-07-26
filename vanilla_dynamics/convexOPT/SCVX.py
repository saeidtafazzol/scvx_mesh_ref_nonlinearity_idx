import numpy as np
import cvxpy as cp
from scipy.integrate import solve_ivp

class CVXProgram:
    def __init__(self, dynamics, N, t_f, T_max, c, x_i, x_f, z_0, init_traj, init_z_map, 
                 trust_region_mode='default', mesh_refinement_enabled=False,
                 # Trust region parameters
                 rho0=0.04, rho1=0.2, rho2=0.7, alpha=1.5, beta=1.5,
                 trust_region_values=None, min_trust_ratio=1e-3,
                 # Optimization parameters
                 tolerance=5e-3, C=5.0, nonlinear_idx_multiplier=0.1,
                 # Constraint parameters
                  mul_min_clamp=0.5, mul_max_clamp=20.0,
                 mul_test_min_clamp=0.1, mul_test_max_clamp=1.0):
        """
        Initialize the CVX program for trajectory optimization.
        
        Parameters:
        -----------
        dynamics : object
            Dynamics object with state_dot functions
        N : int
            Number of discretization points
        t_f : float
            Final time
        T_max : float
            Maximum thrust
        c : float
            Characteristic velocity
        x_i : array
            Initial state
        x_f : array
            Final state
        z_0 : float
            Initial mass parameter
        init_traj : array
            Initial trajectory guess
        init_z_map : array
            Initial mass mapping
        trust_region_mode : str
            Trust region mode ('default' or 'nonlinear_idx')
        mesh_refinement_enabled : bool
            Enable mesh refinement
        rho0, rho1, rho2 : float
            Trust region update thresholds
        alpha, beta : float
            Trust region scaling factors
        trust_region_values : array or None
            Initial trust region values (default creates array of 1e-1)
        min_trust_ratio : float
            Minimum trust region ratio
        tolerance : float
            Convergence tolerance
        C : float
            Penalty parameter
        nonlinear_idx_multiplier : float
            Nonlinearity index multiplier
        s_tolerance : float
            Tolerance for s constraint
        mul_min_clamp, mul_max_clamp : float
            Min/max clamps for multiplier in nonlinear_idx mode
        mul_test_min_clamp, mul_test_max_clamp : float
            Min/max clamps for test multiplier
        """
        self.dynamics = dynamics

        # Trust region parameters
        self.rho0 = rho0
        self.rho1 = rho1
        self.rho2 = rho2
        self.alpha = alpha
        self.beta = beta

        # Configuration flags
        self.mesh_refinement_enabled = mesh_refinement_enabled
        self.trust_region_mode = trust_region_mode

        # Trust region setup
        if trust_region_values is None:
            self.trust_region = np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])
        else:
            self.trust_region = np.array(trust_region_values)
        self.min_trust = self.trust_region * min_trust_ratio

        # Optimization parameters
        self.nonlinear_idx_multiplier = nonlinear_idx_multiplier
        self.tolerance = tolerance
        self.C = C

        # Constraint parameters
        self.mul_min_clamp = mul_min_clamp
        self.mul_max_clamp = mul_max_clamp
        self.mul_test_min_clamp = mul_test_min_clamp
        self.mul_test_max_clamp = mul_test_max_clamp
        # Problem dimensions and variables
        self.N = N
        self.t_f = t_f
        self.T_max = T_max
        self.c = c
        self.n_x = dynamics.n_x
        self.n_u = dynamics.n_u

        self.x_i = x_i
        self.x_f = x_f
        self.z_0 = z_0

        # Initialize trajectories and controls
        self.x_p = init_traj
        self.z_p = init_z_map
        self.u_p = np.zeros((N, dynamics.n_u))
        self.s_p = np.full(self.N - 1, self.t_f)

        self.dt = 1 / (self.N - 1)

        self.previous_nonlinear_cost = None
        self.previous_linear_cost = None

        # Initialize matrices for discretization
        self.A_bar = np.zeros([self.N - 1, self.n_x, self.n_x])
        self.B_bar = np.zeros([self.N - 1, self.n_x, self.n_u])
        self.C_bar = np.zeros([self.N - 1, self.n_x, self.n_u])
        self.d_bar = np.zeros([self.N - 1, self.n_x])
        self.e_bar = np.zeros([self.N - 1, self.n_x])
        self.P = np.zeros([self.N - 1, self.n_x, self.n_x, self.n_x])

        # Vector indices for flat matrices
        x_end = self.n_x
        A_bar_end = x_end + self.n_x * self.n_x
        B_bar_end = A_bar_end + self.n_x * self.n_u
        C_bar_end = B_bar_end + self.n_x * self.n_u
        d_bar_end = C_bar_end + self.n_x
        e_bar_end = d_bar_end + self.n_x
        P_end = e_bar_end + self.n_x**3

        self.x_ind = slice(0, x_end)
        self.A_bar_ind = slice(x_end, A_bar_end)
        self.B_bar_ind = slice(A_bar_end, B_bar_end)
        self.C_bar_ind = slice(B_bar_end, C_bar_end)
        self.d_bar_ind = slice(C_bar_end, d_bar_end)
        self.e_bar_ind = slice(d_bar_end, e_bar_end)
        self.P_ind = slice(e_bar_end, P_end)

        self.V0 = np.zeros((P_end,))
        self.V0[self.A_bar_ind] = np.eye(self.n_x).reshape(-1)

    def piecewise_ode(self, x, t, u0, u1, s):
        u = u0 + (t / self.dt) * (u1 - u0)
        return np.squeeze(self.dynamics.state_dot(x, u, s))

    def piecewise_ode_pmass(self, z, t, u0, u1):
        u = u0 + (t / self.dt) * (u1 - u0)
        return -u / self.c

    def calculate_discretization(self, X, U, S):
        """
        Calculate discretization for given states, inputs and total time.
        """
        for i in range(self.N - 1):
            self.V0[self.x_ind] = X[i, :]
            sol = solve_ivp(
                fun=lambda t, V: self._ode_dVdt(V, t, U[i, :], U[i + 1, :], S[i]),
                t_span=(0, self.dt),
                y0=self.V0,
                method='RK45',
                dense_output=False
            )
            V = sol.y[:, -1]

            # Extract discretized matrices
            Phi = V[self.A_bar_ind].reshape((self.n_x, self.n_x))
            self.A_bar[i, :] = Phi
            self.B_bar[i, :] = Phi @ V[self.B_bar_ind].reshape((self.n_x, self.n_u))
            self.C_bar[i, :] = Phi @ V[self.C_bar_ind].reshape((self.n_x, self.n_u))
            self.d_bar[i, :] = Phi @ V[self.d_bar_ind]
            self.e_bar[i, :] = Phi @ V[self.e_bar_ind]
            self.P[i, :, :] = V[self.P_ind].reshape((self.n_x, self.n_x, self.n_x))

        return self.A_bar, self.B_bar, self.C_bar, self.d_bar, self.e_bar, self.P

    def _ode_dVdt(self, V, t, u_t0, u_t1, s):
        alpha = (self.dt - t) / self.dt
        beta = t / self.dt
        x = V[self.x_ind]
        u = u_t0 + (t / self.dt) * (u_t1 - u_t0)

        Phi_A_xi = np.linalg.inv(V[self.A_bar_ind].reshape((self.n_x, self.n_x)))

        f_subs = self.dynamics.state_dot(x, u, s).squeeze()
        A = self.dynamics.state_dot_x(x, u, s).squeeze()
        B = self.dynamics.state_dot_u(x, u, s).squeeze().transpose()
        d = self.dynamics.state_dot_s(x, u).squeeze()

        phi = V[self.A_bar_ind].reshape((self.n_x, self.n_x))
        e_t = f_subs - A @ x - B @ u - d * s
        psi = V[self.P_ind].reshape((self.n_x, self.n_x, self.n_x))

        dVdt = np.zeros_like(V)
        dVdt[self.x_ind] = f_subs
        dVdt[self.A_bar_ind] = (A @ phi).reshape(-1)
        dVdt[self.B_bar_ind] = (Phi_A_xi @ B).reshape(-1) * alpha
        dVdt[self.C_bar_ind] = (Phi_A_xi @ B).reshape(-1) * beta
        dVdt[self.d_bar_ind] = (Phi_A_xi @ d)
        dVdt[self.e_bar_ind] = Phi_A_xi @ e_t

        dVdt[self.P_ind] = (np.einsum('ace,ed,cb->abd', self.dynamics.state_dot_xx(x, u, s), phi, phi) + 
                           np.einsum('ac,cbd->abd', A, psi)).reshape((self.n_x**3,))

        return dVdt

    def get_prog(self):
        x_dim = self.n_x
        u_dim = self.n_u
        N = self.N
        dt = self.dt
        t_f = self.t_f
        T_max = self.T_max
        c = self.c

        x = cp.Variable((N, x_dim))
        z = cp.Variable(N)
        tau = cp.Variable((N, u_dim))
        tau_len = cp.Variable(N)
        s = cp.Variable(N-1)
        h = cp.Variable((N - 1, x_dim + 1))

        self.x = x
        self.z = z
        self.tau = tau
        self.tau_len = tau_len
        self.s = s
        self.h = h

        objective = self.C * cp.sum(cp.abs(h))

        constraints = []

        A_bar, B_bar, C_bar, d_bar, e_bar, P = self.calculate_discretization(self.x_p, self.u_p, self.s_p)

        nonlinear = np.einsum('il,i->il', np.einsum('ijkl->il', np.abs(P)), 
                             1 / np.einsum('ijk->i', np.abs(A_bar)))
        
        if self.trust_region_mode == 'default':
            mul = np.ones((self.N - 1, self.n_x))
        elif self.trust_region_mode == 'nonlinear_idx':
            mul = np.maximum(np.minimum(self.nonlinear_idx_multiplier / nonlinear, 
                                      self.mul_max_clamp), self.mul_min_clamp)

        constraints.append(cp.abs(x[:-1] - self.x_p[:-1]) <= 
                          np.einsum('i,Ni->Ni', self.trust_region[:-1], mul))

        constraints.append(s >= 0)
        constraints.append(x[0, :] == self.x_i)
        constraints.append(x[-1, :] == self.x_f)
        constraints.append(z[0] == self.z_0)
        constraints.append(cp.norm(tau, 2, axis=1) <= tau_len)
        constraints.append(cp.abs(s - self.s_p) <= self.trust_region[-1])
        
        if not self.mesh_refinement_enabled:
            constraints.append(self.s_p == s)


        for i in range(N - 1):
            objective += dt / 2 * (tau_len[i] + tau_len[i + 1])

            constraints.append(
                x[i + 1, :] == A_bar[i] @ x[i, :] + B_bar[i] @ tau[i, :] +
                C_bar[i] @ tau[i + 1, :] + d_bar[i] * s[i] + e_bar[i] + h[i, :x_dim]
            )
            constraints.append(
                z[i + 1] - z[i] == dt / 2 * (-tau_len[i] - tau_len[i + 1]) / c
            )

            constraints.append(
                tau_len[i] <= self.T_max * np.exp(-self.z_p[i]) * (s[i] - self.s_p[i] * (z[i] - self.z_p[i]))
            )

        constraints.append(
            tau_len[N - 1] <= self.T_max * np.exp(-self.z_p[N - 1]) * (s[N - 2] - self.s_p[N - 2] * (z[N - 1] - self.z_p[N - 1]))
        )
        

        constraints.append(
            cp.sum(s) / (self.N - 1) == self.t_f
        )

        return cp.Minimize(objective), constraints

    def piece_wise_int(self, x_l, z_l, taus, tau_lens, s):
        x_nl = np.zeros_like(x_l)
        z_nl = np.zeros_like(z_l)
        x_nl[0, :] = x_l[0, :]
        z_nl[0] = z_l[0]

        for i in range(self.N - 1):
            # Integrate state dynamics
            sol_x = solve_ivp(
                fun=lambda t, x: self.piecewise_ode(x, t, taus[i, :], taus[i + 1, :], s[i]),
                t_span=(0, self.dt),
                y0=x_l[i, :],
                rtol=1e-15,
                atol=1e-12,
                method='DOP853'
            )
            x_nl[i + 1, :] = sol_x.y[:, -1]
            
            # Integrate mass dynamics
            sol_z = solve_ivp(
                fun=lambda t, z: self.piecewise_ode_pmass(z, t, tau_lens[i], tau_lens[i + 1]),
                t_span=(0, self.dt),
                y0=[z_l[i]],
                rtol=1e-15,
                atol=1e-12,
                method='DOP853'
            )
            z_nl[i + 1] = sol_z.y[0, -1]

        return x_nl, z_nl

    def cal_nonlinear_cost(self):
        if self.x.value is None or self.tau_len.value is None:
            return None

        x_nl, z_nl = self.piece_wise_int(
            self.x.value, self.z.value, self.tau.value, self.tau_len.value , self.s.value
        )
        cost = (
            (self.tau_len.value[1:-1].sum() +
             self.tau_len.value[0] / 2 +
             self.tau_len.value[-1] / 2) * self.dt +
            self.C * np.sum(np.abs(x_nl - self.x.value)) +
            self.C * np.sum(np.abs(z_nl - self.z.value))
        )
        return cost

    def optimize(self, opt_s_flag=False):
        """
        Optimize the trajectory using trust region method.
        
        Parameters:
        -----------
        opt_s_flag : bool
            Optimization flag for s parameter
            
        Returns:
        --------
        tuple : (bool, float, float)
            Convergence flag, linear cost, nonlinear cost
        """
        self.opt_s_flag = opt_s_flag
        obj, const = self.get_prog()
        prob = cp.Problem(obj, const)
        linear_cost = prob.solve(solver=cp.ECOS)

        nonlinear_cost = self.cal_nonlinear_cost()

        self.linear_cost = linear_cost
        self.nonlinear_cost = nonlinear_cost

        actual = None
        predicted = None
        
        if self.previous_nonlinear_cost is not None:
            predicted = self.previous_nonlinear_cost - linear_cost
            actual = self.previous_nonlinear_cost - nonlinear_cost

            ratio = actual / predicted if predicted != 0 else 0

            if ratio < self.rho0:
                self.trust_region /= self.alpha
            else:
                if ratio < self.rho1:
                    self.trust_region /= self.alpha
                elif self.rho2 <= ratio:
                    self.trust_region *= self.beta
                self._update_trajectories()
        else:
            self._update_trajectories()

        self.previous_nonlinear_cost = nonlinear_cost
        self.previous_linear_cost = linear_cost

        # self.trust_region = np.maximum(self.trust_region, self.min_trust)
        
        if predicted is not None:
            converged = (predicted < self.tolerance or 
                        (self.trust_region == self.min_trust).all())
            converged = predicted < self.tolerance
            return converged, linear_cost, nonlinear_cost
        
        return False, linear_cost, nonlinear_cost

    def _update_trajectories(self):
        """Update trajectory variables from optimization results."""
        self.x_p = np.copy(self.x.value)
        self.z_p = np.copy(self.z.value)
        self.u_p = np.copy(self.tau.value)
        self.s_p = np.copy(self.s.value)
