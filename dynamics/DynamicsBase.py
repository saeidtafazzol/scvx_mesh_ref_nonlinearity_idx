
import os
import numpy as np
import casadi as ca

class DynamicsBase:
    def __init__(self, n_x=6, n_u=3):
        self.backend = 'casadi'
        # Define state and control symbols (default sizes, can be overridden)
        self.n_x = n_x
        self.n_u = n_u
        
        # Create temp directory for generated files
        self.temp_dir = "temp"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.states = ca.MX.sym('states', self.n_x)
        self.control = ca.MX.sym('control', self.n_u)
        self.s = ca.MX.sym('s')  # time dilation
        # Common parameter for all dynamics
        self.c = ca.MX.sym('c')

        # Symbolic expressions (CasADi)
        self.state_dot_expr = None
        self.state_dot_x_expr = None
        self.state_dot_u_expr = None
        self.state_dot_s_expr = None
        self.state_dot_xx_expr = None
        
        # Symbolic z_dot expression
        self.z_dot_expr = None

        # Substituted expressions (with parameters)
        self.state_dot_sub    = None
        self.state_dot_s_sub  = None
        self.state_dot_x_sub  = None
        self.state_dot_u_sub  = None
        self.state_dot_xx_sub = None
        
        # Substituted z_dot expression
        self.z_dot_sub = None

        # Dictionary to hold parameter values after set_params is called
        self.params = {}

        # Call the subclass method to populate state_dot_expr
        self.populate_state_dot()

        # Compute derivatives if state_dot_expr is set
        if self.state_dot_expr is not None:
            self.state_dot_x_expr = ca.jacobian(self.state_dot_expr, self.states)
            self.state_dot_u_expr = ca.jacobian(self.state_dot_expr, self.control)
            self.state_dot_s_expr = ca.jacobian(self.state_dot_expr, self.s)
            self.state_dot_xx_expr = ca.jacobian(ca.reshape(self.state_dot_x_expr, (self.n_x * self.n_x, 1)), self.states)
            
        # Compute z_dot_expr = -control_len / c
        control_len = ca.sqrt(self.control[0]**2 + self.control[1]**2 + self.control[2]**2)
        self.z_dot_expr = -control_len / self.c

    def populate_state_dot(self):
        """Subclasses must implement this to populate self.state_dot_expr."""
        raise NotImplementedError("Subclasses must implement populate_state_dot()")

    def configure_params(self, **param_values):
        """Configure the parameter dictionary. Subclasses should override this."""
        params = {}
        if 'c' in param_values:
            params['c'] = param_values['c']  # Use string key instead of CasADi symbol
        return params
    
    def _substitute_params(self, expr):
        # Base class handles the common parameter 'c'
        # Use string key to avoid CasADi symbol boolean issues
        if hasattr(self, 'params') and 'c' in self.params:
            expr = ca.substitute(expr, self.c, self.params['c'])
        return expr

    def set_params(self, **params):
        # Configure parameters with provided values
        self.params = self.configure_params(**params)
        
        # Perform substitutions
        self.state_dot_sub    = self._substitute_params(self.state_dot_expr)
        self.state_dot_s_sub  = self._substitute_params(self.state_dot_s_expr)
        self.state_dot_x_sub  = self._substitute_params(self.state_dot_x_expr)
        self.state_dot_u_sub  = self._substitute_params(self.state_dot_u_expr)
        self.state_dot_xx_sub = self._substitute_params(self.state_dot_xx_expr)
        self.z_dot_sub        = self._substitute_params(self.z_dot_expr)

    def _compile_function(self, func_name, expr, filename_base, use_state=True, use_control=True, use_s=True):
        inputs = []
        if use_state:
            inputs.append(self.states)
        if use_control:
            inputs.append(self.control)
        if use_s:
            inputs.append(self.s)
        
        func_aux = ca.Function(func_name, inputs, [expr])
        
        # Save current directory and change to temp directory
        current_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        try:
            c_filename = f"{filename_base}.c"
            so_filename = f"{filename_base}.so"
            func_aux.generate(c_filename)
            os.system(f"gcc -fPIC -shared {c_filename} -o {so_filename}")
            # Return to original directory and create external function with full path
            os.chdir(current_dir)
            return ca.external(func_name, os.path.join(self.temp_dir, so_filename))
        except Exception as e:
            # Make sure we return to original directory even if there's an error
            os.chdir(current_dir)
            raise e

    def compile(self):
        if self.state_dot_sub is None:
            print("Please call set_params() first.")
            return
        self.state_dot_compiled  = self._compile_function("state_dot", self.state_dot_sub, "cr3bp_state_dot")
        self.state_dot_s_compiled = self._compile_function("state_dot_s", self.state_dot_s_sub, "cr3bp_state_dot_s", use_s=False)
        self.state_dot_x_compiled = self._compile_function("state_dot_x", self.state_dot_x_sub, "cr3bp_state_dot_x")
        self.state_dot_u_compiled = self._compile_function("state_dot_u", self.state_dot_u_sub, "cr3bp_state_dot_u")
        self.state_dot_xx_compiled = self._compile_function("state_dot_xx", self.state_dot_xx_sub, "cr3bp_state_dot_xx")
        self.z_dot_compiled = self._compile_function("z_dot", self.z_dot_sub, "z_dot", use_state=False, use_s=False)

    # Wrapper functions for numerical evaluation
    def state_dot(self, state, control, s):
        state_ca = ca.DM(state)
        control_ca = ca.DM(control)
        s_ca = ca.DM(s)
        result = self.state_dot_compiled(state_ca, control_ca, s_ca)
        return np.array(result.full()).flatten()
    
    def state_dot_s(self, state, control):
        state_ca = ca.DM(state)
        control_ca = ca.DM(control)
        result = self.state_dot_s_compiled(state_ca, control_ca)
        return np.array(result.full()).flatten()

    def state_dot_x(self, state, control, s):
        state_ca = ca.DM(state)
        control_ca = ca.DM(control)
        s_ca = ca.DM(s)
        result = self.state_dot_x_compiled(state_ca, control_ca, s_ca)
        return np.array(result.full()).reshape((self.n_x, self.n_x))
    
    def state_dot_u(self, state, control, s):
        state_ca = ca.DM(state)
        control_ca = ca.DM(control)
        s_ca = ca.DM(s)
        result = self.state_dot_u_compiled(state_ca, control_ca, s_ca)
        return np.array(result.full()).reshape((self.n_u, self.n_x))
    
    def state_dot_xx(self, state, control, s):
        state_ca = ca.DM(state)
        control_ca = ca.DM(control)
        s_ca = ca.DM(s)
        result = self.state_dot_xx_compiled(state_ca, control_ca, s_ca)
        return np.array(result.full()).reshape((self.n_x, self.n_x, self.n_x))

    def z_dot(self, control):
        control_ca = ca.DM(control)
        result = self.z_dot_compiled(control_ca)
        return np.array(result.full()).flatten()
