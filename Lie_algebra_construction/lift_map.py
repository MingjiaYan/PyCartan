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
    ):
        self.K = dimension
        self.domain = domain
        self.dt = dt
        
        # Initialize coefficients ξⁱₙ(t) as a 2D array [K × n_modes]
        self.coefficients = np.zeros((self.K, self.domain.n_modes))
        
    def get_basis_functions(self) -> List[Callable]:
        """Return the spatial basis functions ψₙ based on domain type."""
        if self.domain.type == DomainType.CIRCLE:
            return [
                lambda u, n=n: np.sin(n * u) if n > 0 else np.cos(n * u)
                for n in range(self.domain.n_modes)
            ]
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
        basis_fns = self.get_basis_functions()
        result = np.zeros(self.K)
        
        for i in range(self.K):
            for n, psi in enumerate(basis_fns):
                result[i] += self.coefficients[i, n] * psi(u)
                
        return result
    
    def set_coefficients(self, coeff_matrix: np.ndarray):
        """Set the coefficients ξⁱₙ directly."""
        assert coeff_matrix.shape == (self.K, self.domain.n_modes)
        self.coefficients = coeff_matrix.copy()
    
    def time_derivative(self) -> np.ndarray:
        """
        Compute time derivative of coefficients using finite differencing.
        This would be used in time evolution schemes.
        """
        # This is a placeholder for actual finite differencing implementation
        # You would need past values of coefficients to compute this properly
        pass

# Example usage for circle domain
circle_domain = Domain(
    type=DomainType.CIRCLE,
    bounds=(0, 2*np.pi),
    n_modes=5
)

# Create a Lie algebra element with K=3 generators
xi = LieAlgebraElement(
    dimension=3,
    domain=circle_domain,
    dt=0.01
)

# Set some example coefficients
example_coeffs = np.random.rand(3, 5)  # 3 generators × 5 modes
xi.set_coefficients(example_coeffs)

# Evaluate at a specific point and time
u_test = np.pi/4
t_test = 5.0
values = xi.evaluate(u_test, t_test)
