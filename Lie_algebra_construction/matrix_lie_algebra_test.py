import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from algebra_element import DomainType, Domain, LieAlgebraElement


class MatrixLieAlgebraTests:
    @staticmethod
    def se2_generators() -> List[np.ndarray]:
        """
        Returns the generators of se(2) in conventional representation.
        TODO: VERIFY OUR SPECIFIC NOTATIONS.
        X: infinitesimal x-translation
        Y: infinitesimal y-translation
        R: infinitesimal rotation.
        """

        # Generator for x-translation
        X = np.array([
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0]
        ])

        # Generator for y-translation
        Y = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])

        # Generator for rotation
        R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ])

        return [X, Y, R]
    
    @staticmethod
    def compute_commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """"Compute the matrix commutator [A, B] = AB - BA."""
        return A @ B - B @ A
    
    def test_se2_commutation_relations(self, coeffs:np.ndarray,
                                       u: float, t: float) -> Tuple[bool, dict]:
        """
        Test:
        1. if the constructed Lie algebra element in our prev formulation satisfies se(2) relations.
        The non-zero commutators are:
        [R, X] = Y
        [R, Y] = -X
        [X, Y] = 0;
        2. Commutation relations of our constructed Lie algebra element.
        """
        generators = self.se2_generators()
        X, Y, R = generators

        # Construct the matrix representation of our element
        # xi = a^1 X + a^2 Y + a^3 R, where a^i == xi^i are our coefficients.
        xi = sum(c * T for c, T in zip(coeffs, generators))
        print("\nConstructed Lie algebra element:")
        print(xi)

        # Test the commutator relations one by one
        results = {}

        # [R, X] = Y
        commutator_RX = self.compute_commutator(R, X)
        error_RX = np.max(np.abs(commutator_RX - Y))
        results['[R, X] = Y'] = error_RX

        # [R, Y] = -X
        commutator_RY = self.compute_commutator(R, Y)
        error_RY = np.max(np.abs(commutator_RY + X))
        results['[R, Y] = -X'] = error_RY

        # [X, Y] = 0
        commutator_XY = self.compute_commutator(X, Y)
        error_XY = np.max(np.abs(commutator_XY))
        results['[X, Y] = 0'] = error_XY

        # Test passed if all errors are small
        passed = all(error < 1e-10 for error in results.values())

        return passed, results

    def test_se2_action(self, coeffs: np.ndarray) -> Tuple[bool, float]:
        """
        Test if the element yields the correct infinitesimal action on points (x, y).
        The action of an se(2) algebra valued element xi on a point (x, y) in R^2 is:
        ξ dot (x, y) = (a^1 - a^3 y, a^2 + a^3 x)
        """
        generators = self.se2_generators()
        xi = sum(c * T for c, T in zip(coeffs, generators))

        # Test points
        test_points = [
            np.array([1, 2, 1]),   # coord in homogeneous space is (1, 2)
            np.array([0, 1, 1]),   # coord in homogeneous space is (0, 1)
            np.array([-3, 5, 1]),  # coord in homogeneous space is (-3, 5)
        ]

        max_error = 0.0
        for p in test_points:
            # Matrix action
            xi_p = xi @ p
            
            x, y = p[0], p[1]  # Extract x and y coordinates
            expected = np.array([
                coeffs[0] - coeffs[2]*y,  # dx/dt
                coeffs[1] + coeffs[2]*x,  # dy/dt
                0
            ])

            error = np.max(np.abs(xi_p - expected))
            max_error = max(max_error, error)
        
        passed = max_error < 1e-10
        return passed, max_error
    
    def compute_structure_constants(self) -> np.ndarray:
        """
        Compute all structure constants c^k_{ij} for se(2).
        Return a 3x3x3 array where c[k,i,j] = c^k_{ij}
        """
        generators = self.se2_generators()
        n = len(generators)
        c = np.zeros((n, n, n))
        
        for i in range(n):
            for j in range(n):
                # Compute [e_i,e_j]
                comm = self.compute_commutator(generators[i], generators[j])
                
                # Express result in terms of basis elements
                for k in range(n):
                    # The structure constant c^k_{ij} is the coefficient of e_k
                    # in the expansion of [e_i,e_j]
                    c[k,i,j] = np.trace(generators[k].T @ comm) / np.trace(generators[k].T @ generators[k])
        
        return c

    def test_structure_constants(self) -> Tuple[bool, dict]:
        """
        Test if computed structure constants match theoretical values.
        Returns (passed, results).
        """
        # Compute actual structure constants
        c = self.compute_structure_constants()
        
        # Expected non-zero structure constants
        expected = {
            'c^2_{13}': (1, 2, 0, 1.0),  # [X,R] = Y
            'c^3_{12}': (0, 2, 1, -1.0),   # [Y,R] = -X
        }
        
        results = {}
        for name, (k,i,j,expected_value) in expected.items():
            actual_value = c[k,i,j]
            error = abs(actual_value - expected_value)
            results[name] = {'expected': expected_value,
                           'computed': actual_value,
                           'error': error}
        
        # All other structure constants should be zero
        max_other = 0.0
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    if not any((k==k_,i==i_,j==j_) for k_,i_,j_,_ in expected.values()):
                        max_other = max(max_other, abs(c[k,i,j]))
        results['other'] = {'expected': 0.0,
                          'max_error': max_other}
        
        # Print the full structure constant tensor for verification
        print("\nStructure Constants c^k_{ij}:")
        print("-----------------------------")
        for k in range(3):
            print(f"\nk = {k}:")
            print(c[k])
        
        # Test if all values within tolerance
        tol = 1e-10
        passed = all(r['error'] < tol for r in results.values() if 'error' in r)
        passed = passed and max_other < tol
        
        return passed, results

# Run the tests
def run_structure_tests():
    tester = MatrixLieAlgebraTests()
    passed, results = tester.test_structure_constants()
    
    print("\nStructure Constant Test Results:")
    print("--------------------------------")
    print(f"Overall: {'passed' if passed else 'failed'}\n")
    
    for name, result in results.items():
        if name != 'other':
            print(f"{name}:")
            print(f"  Expected: {result['expected']}")
            print(f"  Computed: {result['computed']}")
            print(f"  Error: {result['error']}\n")
        else:
            print("All other structure constants:")
            print(f"  Maximum deviation from zero: {result['max_error']}")


run_structure_tests()


# Example:
def run_se2_tests(xi: LieAlgebraElement, u_test: float, t_test: float):
    tester = MatrixLieAlgebraTests()

    # Get the coefficients at test points
    coeffs = xi.evaluate(u_test, t_test)

    # Test commutation relations
    passed_commutator, results = tester.test_se2_commutation_relations(coeffs, u_test, t_test)
    print("\nCommutation relations test:")
    print(f"Overall: {'passed' if passed_commutator else 'failed'}")
    for relation, error in results.items():
        print(f"{relation}: error = {error}")
    
    # Test se(2) action on R^2
    passed_action, error_action = tester.test_se2_action(coeffs)
    print("\nAction on points test:")
    print(f"{'Passed' if passed_action else 'Failed'} with max error {error_action}")


# Test with some example coefficients
circle_domain = Domain(
    type=DomainType.CIRCLE,
    bounds=(0, 2*np.pi),
    n_modes=2  # 2
)

xi = LieAlgebraElement(
    dimension=3,  # se(2) is 3-dimensional: 2 translations + 1 rotation
    domain=circle_domain,
    dt=0.01,
    n_timesteps=100
)

# Expected dimensions
print(f"K (dimension of se(2)): {xi.K}")
print(f"total_coeffs: {xi.total_coeffs}")
print(f"Expected coefficient shape: ({xi.K}, {xi.total_coeffs})")

# Actual shape
test_coeffs = np.random.rand(3, 5)  # np.random.rand(3, 5)
print(f"Actual coefficient shape: {test_coeffs.shape}")

# Then calculate total_coeffs to verify our logic:
n_modes = circle_domain.n_modes
total_coeffs_calc = 2 * n_modes + 1
print(f"\nCalculation check:")
print(f"n_modes: {n_modes}")
print(f"2 * n_modes + 1 = {total_coeffs_calc}")

# Set some test coefficients
xi.set_coefficients(0.0, test_coeffs)

# Run tests
run_se2_tests(xi, u_test=np.pi/4, t_test=0.0)



# -------- flow lines visualisation ---------------------

def visualise_se2_flow(xi_matrix, grid_size=10, scale=0.5):
    """
    Visualise the flow field of an SE(2) Lie algebra element.
    
    Parameters:
    -----------
    xi_matrix : array_like
        3x3 matrix representing the SE(2) Lie algebra element
    grid_size : int
        Number of points along each axis
    scale : float
        Scale factor for velocity arrows
    """
    # Create grid points
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Extract coefficients from the matrix
    a1 = xi_matrix[0, 2]  # x-translation
    a2 = xi_matrix[1, 2]  # y-translation
    a3 = xi_matrix[1, 0]  # rotation
    
    # Compute velocity field
    U = a1 - a3 * Y  # x-component of velocity
    V = a2 + a3 * X  # y-component of velocity
    
    # Create figure
    plt.figure(figsize=(6, 6))
    
    # Plot velocity field
    plt.quiver(X, Y, U, V, scale=1/scale, width=0.003)
    
    # Add grid and axis lines
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Set equal aspect ratio and limits
    plt.axis('equal')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    
    # Add title and labels
    plt.title('SE(2) Flow Field Visualisation', pad=10)
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Add text showing the coefficients
    coeff_text = f'Coefficients:\n' \
                 f'Translation: ({a1:.3f}, {a2:.3f})\n' \
                 f'Rotation: {a3:.3f}'
    plt.text(2.6, 0, coeff_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    return plt.gcf()

# Example
xi_matrix = np.array([
    [ 0.0, -0.4858853, 0.33703326],
    [ 0.4858853, 0.0, 0.68241567],
    [ 0.0, 0.0, 0.0]
])

# Create visualisation
fig = visualise_se2_flow(xi_matrix, grid_size=10, scale=0.5)


def pure_motions():
    # Pure translation in x
    xi_trans_x = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    
    # Pure translation in y
    xi_trans_y = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0]
    ])
    
    # Pure rotation
    xi_rot = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    
    # Create subplot figure
    plt.figure(figsize=(6, 6))
    
    # Plot pure x-translation
    plt.subplot(131)
    visualise_se2_flow(xi_trans_x, grid_size=8, scale=0.3)
    plt.title('Pure X-Translation')
    
    # Plot pure y-translation
    plt.subplot(132)
    visualise_se2_flow(xi_trans_y, grid_size=8, scale=0.3)
    plt.title('Pure Y-Translation')
    
    # Plot pure rotation
    plt.subplot(133)
    visualise_se2_flow(xi_rot, grid_size=8, scale=0.3)
    plt.title('Pure Rotation')
    
    plt.tight_layout()
    return plt.gcf()

# Pure motions visualisation
fig_pure = pure_motions()

# Show both figures
plt.show()



# --------- Visualise time evolution of the flow and the first modes ----------------------

def visualise_flow_snapshots(coeffs_history, t_points):
    """Create snapshots of the flow field at different times."""
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = axes.flatten()
    
    for idx, t in enumerate([0, 250, 500, 750]):  # Indices into coeffs_history
        ax = axes[idx]
        
        # Get coefficients at this time
        coeffs = coeffs_history[idx]
        
        # Create corresponding se(2) matrix
        xi_matrix = np.array([
            [0.0, -coeffs[2], coeffs[0]],
            [coeffs[2], 0.0, coeffs[1]],
            [0.0, 0.0, 0.0]
        ])
        
        # Create grid
        grid_size = 8
        x = np.linspace(-2, 2, grid_size)
        y = np.linspace(-2, 2, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Compute velocity field
        U = coeffs[0] - coeffs[2] * Y
        V = coeffs[1] + coeffs[2] * X
        
        # Plot
        ax.quiver(X, Y, U, V, scale=2)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', alpha=0.3)
        ax.axvline(x=0, color='k', alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title(f't = {t_points[idx]:.2f}')
    
    plt.tight_layout()
    return fig


def test_time_evolution(xi, t_range):
    """
    Test and visualise the time evolution of SE(2) coefficients.
    
    Parameters:
    -----------
    xi : LieAlgebraElement
        The Lie algebra element with time-dependent coefficients
    t_range : array_like
        Array of time points to evaluate
    """
    # Store coefficients at each time
    coeffs_history = []
    
    # Evaluate at a fixed spatial point (e.g. u = π/4)
    u_test = np.pi/4
    
    for t in t_range:
        coeffs = xi.get_coefficients(t)[:,0]  # Get first modes only
        coeffs_history.append(coeffs)

    # for t in t_range:
    #     # Get all coefficients at this time and spatial point
    #     coeffs = xi.evaluate(u_test, t)  # sum over all modes
    #     coeffs_history.append(coeffs)
    
    coeffs_history = np.array(coeffs_history)
    
    # Plot the evolution
    plt.figure(figsize=(12, 8))
    
    # Plot each component
    plt.subplot(311)
    plt.plot(t_range, coeffs_history[:, 0], 'b-', label='Translation X')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f'Time Evolution of SE(2) Coefficients (First mode constant terms)')
    plt.ylabel('X Coefficient')
    
    plt.subplot(312)
    plt.plot(t_range, coeffs_history[:, 1], 'g-', label='Translation Y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylabel('Y Coefficient')
    
    plt.subplot(313)
    plt.plot(t_range, coeffs_history[:, 2], 'r-', label='Rotation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('R Coefficient')
    
    plt.tight_layout()
    
    return coeffs_history

# Create test data with time-dependent coefficients
circle_domain = Domain(
    type=DomainType.CIRCLE,
    bounds=(0, 2*np.pi),
    n_modes=2
)

xi = LieAlgebraElement(
    dimension=3,  # se(2) dimension
    domain=circle_domain,
    dt=0.01,
    n_timesteps=200
)

# Set time-dependent coefficients
t_points = np.arange(0, 1, 0.01)
for t in t_points:
    # Create coefficients that vary sinusoidally with time
    coeffs = np.zeros((3, 5))  # 3 components × 5 coefficients (coeffs for const. term 1/\sqrt{2\pi}, \cos(u), \sin(u), \cos(2u), \sin(2u))
    
    # Constant term varies with time
    coeffs[0, 0] = np.sin(2*np.pi*t)  # X translation
    coeffs[1, 0] = np.cos(2*np.pi*t)  # Y translation
    coeffs[2, 0] = 0.5*np.sin(4*np.pi*t)  # Rotation
    
    # Add some variation in the other modes (not used for this plot)
    coeffs[0, 1] = 0.3*np.sin(3*np.pi*t)
    coeffs[1, 1] = 0.3*np.cos(3*np.pi*t)
    coeffs[2, 1] = 0.2*np.sin(5*np.pi*t)
    
    xi.set_coefficients(t, coeffs)


coeffs_history = test_time_evolution(xi, t_points)

# Create flow snapshots
flow_fig = visualise_flow_snapshots(coeffs_history, t_points)

plt.show()
