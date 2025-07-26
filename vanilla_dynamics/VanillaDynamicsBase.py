import os
import numpy as np
import casadi as ca

class VanillaDynamicsBase:
    """
    Base class for vanilla dynamics without convex optimization features.
    Simple dynamics with state_dot, m_dot, and compilation.
    """
    def __init__(self, n_x=6, n_u=3):
        self.n_x = n_x
        self.n_u = n_u
        
        # Create temp directory for generated files
        self.temp_dir = "temp"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        # Define symbolic variables for compilation
        self.states = ca.MX.sym('states', self.n_x)
        self.control = ca.MX.sym('control', self.n_u)
        self.mass = ca.MX.sym('mass')
        
        # Parameters dictionary
        self.params = {}
        
        # Symbolic expressions (to be populated by subclasses)
        self.state_dot_expr = None
        self.m_dot_expr = None
        
        # Compiled functions
        self.state_dot_func = None
        self.m_dot_func = None
        
        # Call subclass to populate expressions
        self.populate_expressions()
        
    def populate_expressions(self):
        """Subclasses must implement this to populate symbolic expressions."""
        raise NotImplementedError("Subclasses must implement populate_expressions()")
        
    def set_params(self, **params):
        """Set parameters for the dynamics."""
        self.params.update(params)
        
    def _substitute_params(self, expr):
        """Substitute parameter values into symbolic expression."""
        for param_name, param_value in self.params.items():
            if param_name in ['mu', 'c']:  # Add other parameters as needed
                expr = ca.substitute(expr, ca.MX.sym(param_name), param_value)
        return expr
        
    def _compile_function(self, func_name, expr, filename_base, inputs):
        """Compile a CasADi function to shared library."""
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
        """Compile the dynamics functions."""
        if self.state_dot_expr is None or self.m_dot_expr is None:
            print("Please ensure expressions are populated first.")
            return
            
        # Substitute parameters
        state_dot_sub = self._substitute_params(self.state_dot_expr)
        m_dot_sub = self._substitute_params(self.m_dot_expr)
        
        # Compile functions
        self.state_dot_func = self._compile_function(
            "state_dot_func", 
            state_dot_sub, 
            "vanilla_state_dot",
            [self.states, self.control, self.mass]
        )
        
        self.m_dot_func = self._compile_function(
            "m_dot_func", 
            m_dot_sub, 
            "vanilla_m_dot",
            [self.control]
        )
        
    def state_dot(self, state, control, mass):
        """
        Compute state derivative using compiled function.
        
        Parameters:
        -----------
        state : array_like, shape (n_x,)
            Current state vector
        control : array_like, shape (n_u,)
            Control vector
        mass : float
            Current mass
            
        Returns:
        --------
        ndarray, shape (n_x,)
            State derivative
        """
        if self.state_dot_func is None:
            raise RuntimeError("Please call compile() first.")
            
        state_ca = ca.DM(state)
        control_ca = ca.DM(control)
        mass_ca = ca.DM(mass)
        result = self.state_dot_func(state_ca, control_ca, mass_ca)
        return np.array(result.full()).flatten()
        
    def m_dot(self, control):
        """
        Compute mass derivative using compiled function.
        
        Parameters:
        -----------
        control : array_like, shape (n_u,)
            Control vector
            
        Returns:
        --------
        float
            Mass derivative
        """
        if self.m_dot_func is None:
            raise RuntimeError("Please call compile() first.")
            
        control_ca = ca.DM(control)
        result = self.m_dot_func(control_ca)
        return float(result.full().flatten()[0])
