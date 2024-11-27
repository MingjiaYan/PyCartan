import numpy as np
from enum import Enum
from typing import Callable, List, Union, Optional
from dataclasses import dataclass


class DomainType(Enum):
    CIRCLE = "S1"  # S¹ domain
    INTERVAL = "I"  # Bounded interval


@dataclass
class Domain:
    type: DomainType
    bounds: tuple[float, float]  # For interval [-1,1] or circle [0, 2π]
    n_modes: int  # Number of modes in spectral expansion


class LieAlgebraElement:
    def __init__(
        self,
        dimension: int,  # Dimension K of the Lie algebra
        domain: Domain,
        dt: float,  # Time step for finite differencing
        n_timesteps: int,  # Number of timesteps to store
    ):
        self.K = dimension
        self.domain = domain
        self.dt = dt
        self.n_timesteps = n_timesteps

        # E.g. Fourier series: 2*n_modes + 1 coefficient per dimension
        self.total_coeffs = 2 * self.domain.n_modes + 1 if domain.type == DomainType.CIRCLE else self.domain.n_modes
        
        # Initialise coefficients ξ^i(t) as a 3D array [time x K × n_modes]
        self.coefficients = np.zeros((n_timesteps, self.K, self.total_coeffs))
        self.current_time_idx = 0
        
    def basis_functions(self) -> List[Callable]:
        """Return the spatial basis functions ψ_n based on domain type."""
        if self.domain.type == DomainType.CIRCLE:
            
            basis_fns = []
            # Constant term (n=0)
            basis_fns.append(lambda u: np.ones_like(u) / np.sqrt(2*np.pi))
            
            # For each n>0, a superposition of sin and cos
            for n in range(1, self.domain.n_modes + 1):
                basis_fns.append(lambda u, n=n: np.cos(n*u) / np.sqrt(np.pi))
                basis_fns.append(lambda u, n=n: np.sin(n*u) / np.sqrt(np.pi))
                
            return basis_fns

        elif self.domain.type == DomainType.INTERVAL:
            return [
                lambda u, n=n: np.cos(n * np.arccos(u))  # Chebyshev polynomials
                for n in range(self.domain.n_modes)
            ]
        
    def evaluate(self, u: float, t: float) -> np.ndarray:
        """
        Evaluate ξ at spatial point u and time t.
        Returns a vector of length K (one component per basis element).
        """
        basis_fns = self.basis_functions()
        result = np.zeros(self.K)

        # Convert time to index
        t_idx = int(round(t / self.dt))
        t_idx = max(0, min(t_idx, self.n_timesteps - 1))  # Ensure within bounds
        
        for i in range(self.K):
            for n, psi in enumerate(basis_fns):
                result[i] += self.coefficients[t_idx, i, n] * psi(u)
                
        return result
    
    def set_coefficients(self, t: float, coeff_matrix: np.ndarray):
        """Set the coefficients ξ^i_n at a specific time.
           A setter function that WRITES data to the coefficient array.
        """
        assert coeff_matrix.shape == (self.K, self.total_coeffs)
        t_idx = int(round(t / self.dt))
        if 0 <= t_idx < self.n_timesteps:
            self.coefficients[t_idx] = coeff_matrix.copy()
        else:
            raise ValueError(f"Time {t} is outside the stored time range.")
    
    def get_coefficients(self, t: float) -> np.ndarray:
        """
        Get the coefficients ξ^i_n at a specific time.
        A getter function that READS data from the coefficient array.
        """
        t_idx = int(round(t / self.dt))
        if 0 <= t_idx < self.n_timesteps:
            return self.coefficients[t_idx]
        else:
            raise ValueError(f"Time {t} is outside the stored time range.")
    
    def time_derivative(self) -> np.ndarray:
        """
        Compute time derivative of coefficients using finite differencing.
        This would be used in time evolution schemes.
        """
        # This is a placeholder for actual finite differencing implementation
        # You would need past values of coefficients to compute this properly
        pass


# Example:
# circle_domain = Domain(
#     type=DomainType.CIRCLE,
#     bounds=(0, 2*np.pi),
#     n_modes=5
# )

# # Create a Lie algebra element with K=3 generators
# xi = LieAlgebraElement(
#     dimension=3,
#     domain=circle_domain,
#     dt=0.01,
#     n_timesteps=1000  # Store 1000 timesteps
# )

# # Set some example coefficients
# example_coeffs = np.random.rand(3, 5)  # 3 generators × 5 modes
# xi.set_coefficients(example_coeffs)

# # Set coefficients at different times
# t0 = 0.0
# t1 = 0.5
# coeffs_t0 = np.random.rand(3, 5)
# coeffs_t1 = np.random.rand(3, 5)

# xi.set_coefficients_at_time(t0, coeffs_t0)
# xi.set_coefficients_at_time(t1, coeffs_t1)

# # Evaluate at different space-time points
# u_test = np.pi/4
# values_t0 = xi.evaluate(u_test, t0)
# values_t1 = xi.evaluate(u_test, t1)


class LieAlgebraElementVisualizer:
    @staticmethod
    def explain_coefficient(t_idx: int, i: int, n: int, dt: float, n_modes: int) -> str:
        """Explain what a specific coefficient represents."""
        time = t_idx * dt
        
        # Determine which basis function this coefficient corresponds to
        if n == 0:
            basis = "constant term (1/√(2π))"
        else:
            mode = (n + 1) // 2
            if n % 2 == 1:
                basis = f"cos({mode}u)/√π"
            else:
                basis = f"sin({mode}u)/√π"
                
        return f"ξ{i+1}(t={time:.2f}) coefficient for {basis}"

    @staticmethod
    def print_coefficient_structure(n_timesteps: int, K: int, n_modes: int, dt: float):
        """Print a human-readable explanation of the coefficient structure."""
        total_coeffs = 2 * n_modes + 1
        
        print(f"Coefficient array structure ({n_timesteps} × {K} × {total_coeffs}):")
        print("\nExample entries:")
        
        # Print a few example entries
        for t in range(min(2, n_timesteps)):
            for i in range(min(2, K)):
                for n in range(min(3, total_coeffs)):
                    print(f"coefficients[{t}, {i}, {n}]: {LieAlgebraElementVisualizer.explain_coefficient(t, i, n, dt, n_modes)}")
                print()


# Example:
visualizer = LieAlgebraElementVisualizer()
visualizer.print_coefficient_structure(
    n_timesteps=4,
    K=3,
    n_modes=2,
    dt=0.01
)
