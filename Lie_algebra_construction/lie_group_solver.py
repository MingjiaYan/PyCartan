import numpy as np
from scipy.linalg import expm
from typing import Callable, Optional
from dataclasses import dataclass
from algebra_element import LieAlgebraElement, DomainType, Domain


@dataclass
class LieGroupIntegrator:
    """
    Implements Munthe-Kaas method for integrating Lie group ODEs of the form dg = gξ
    where g is an element of a matrix Lie group and ξ is an element of its Lie algebra.
    """
    dimension: int  # Dimension of matrices (n×n)
    dt: float  # Time step
    order: int = 4  # Order of Runge-Kutta method (default RK4)
    
    def vector_to_matrix_so3(self, v: np.ndarray) -> np.ndarray:
        """Convert vector representation to matrix representation for so(3)."""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def adjoint_action(self, u_mat: np.ndarray, x_mat: np.ndarray) -> np.ndarray:
        """Compute the adjoint action ad_u(x) = [u,x] = ux - xu."""
        return u_mat @ x_mat - x_mat @ u_mat
    
    def dexpinv(self, u: np.ndarray) -> np.ndarray:
        """
        Compute the inverse of the derivative of exp at u.
        Uses truncated Baker-Campbell-Hausdorff formula.
        """
        n = self.dimension
        n2 = n * n
        
        # If u is close to zero, return identity
        if np.allclose(u, 0):
            return np.eye(n2)
        
        # Reshape u to matrix form for adjoint calculations
        u_mat = u.reshape((n, n))
        
        # Initialize result matrix
        B = np.zeros((n2, n2))
        
        # Create basis matrices
        basis_matrices = []
        for i in range(n2):
            ei = np.zeros(n2)
            ei[i] = 1
            basis_matrices.append(ei.reshape(n, n))
        
        # Compute action on each basis element
        for i, ei_mat in enumerate(basis_matrices):
            result = ei_mat.copy()
            term = ei_mat.copy()
            
            # Compute terms up to order 4
            for k in range(1, 5):
                term = self.adjoint_action(u_mat, term)
                result += (-1)**(k-1) * term / k
                
            B[:, i] = result.reshape(n2)
        
        return B
    
    def rk4_step(self, 
                 xi: LieAlgebraElement,
                 g: np.ndarray, 
                 t: float) -> np.ndarray:
        """
        Perform one step of RK4 integration in the Lie algebra.
        Maps back to Lie group using exponential map.
        """
        dt = self.dt
        n = self.dimension
        
        # Get current Lie algebra element and convert to matrix form
        v = xi.evaluate(t, t)  # This returns a 3D vector for so(3)
        v_mat = self.vector_to_matrix_so3(v)
        v_flat = v_mat.reshape(n*n)
        
        # RK4 in the Lie algebra
        k1 = dt * (self.dexpinv(np.zeros(n*n)) @ v_flat)
        k2 = dt * (self.dexpinv(k1/2) @ v_flat)
        k3 = dt * (self.dexpinv(k2/2) @ v_flat)
        k4 = dt * (self.dexpinv(k3) @ v_flat)
        
        # Combine steps
        u = (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Map back to Lie group using matrix exponential
        return g @ expm(u.reshape((n, n)))
    
    def integrate(self, 
                 xi: LieAlgebraElement,
                 g0: np.ndarray,
                 t_span: tuple[float, float],
                 n_steps: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Integrate the Lie group ODE dg = gξ from t_start to t_end.
        """
        t_start, t_end = t_span
        if n_steps is None:
            n_steps = int((t_end - t_start) / self.dt)
        
        # Initialize arrays for solution
        t_points = np.linspace(t_start, t_end, n_steps+1)
        g_trajectory = np.zeros((n_steps+1, self.dimension, self.dimension))
        g_trajectory[0] = g0
        
        # Integrate
        g = g0.copy()
        for i in range(n_steps):
            g = self.rk4_step(xi, g, t_points[i])
            g_trajectory[i+1] = g
            
        return t_points, g_trajectory

def test_integrator():
    """
    Test the integrator on SO(3) with a simple time-dependent ξ(t).
    """
    # Create a domain for time-varying Lie algebra element
    domain = Domain(
        type=DomainType.INTERVAL,
        bounds=(0, 1),
        n_modes=5
    )
    
    # Create a time-dependent Lie algebra element (3×3 skew-symmetric matrices)
    xi = LieAlgebraElement(
        dimension=3,  # This represents the 3D vector for so(3)
        domain=domain,
        dt=0.01,
        n_timesteps=1000
    )
    
    # Set some time-varying coefficients (rotation around z-axis)
    t = np.linspace(0, 1, 1000)
    for i, t_i in enumerate(t):
        coeffs = np.zeros((3, 5))  # 3 components, 5 modes
        coeffs[0, 0] = -np.sin(2*np.pi*t_i)  # ω_x
        coeffs[1, 0] = np.cos(2*np.pi*t_i)   # ω_y
        coeffs[2, 0] = 0.5                    # ω_z
        xi.set_coefficients(t_i, coeffs)
    
    # Create integrator
    integrator = LieGroupIntegrator(dimension=3, dt=0.01)
    
    # Initial condition (identity in SO(3))
    g0 = np.eye(3)
    
    # Integrate
    t_points, g_trajectory = integrator.integrate(
        xi=xi,
        g0=g0,
        t_span=(0, 1),
        n_steps=100
    )
    
    return t_points, g_trajectory

if __name__ == "__main__":
    t, g = test_integrator()
    print("Integration complete!")
    print("Final rotation matrix:")
    print(g[-1])
    
    # Verify the result is still in SO(3)
    final_g = g[-1]
    print("\nVerifying SO(3) properties:")
    print("Det(g) = ", np.linalg.det(final_g))
    print("g^T g = I error: ", np.max(np.abs(final_g.T @ final_g - np.eye(3))))
