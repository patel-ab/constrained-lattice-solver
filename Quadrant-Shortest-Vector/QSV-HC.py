import numpy as np
import time

def generate_random_lattice(dimension, size):
    """
    Generate a random lattice with the given dimension and size.
    """
    basis = np.random.randint(-size, size, size=(dimension, dimension))
    while np.linalg.matrix_rank(basis) < dimension:
        basis = np.random.randint(-size, size, size=(dimension, dimension))
    return basis

def objective_function(vector):
    """
    Objective function to minimize: the Euclidean norm of the vector.
    """
    return np.linalg.norm(vector)

def hill_climbing_quadrant_svp(basis, max_iterations=1000, step_size=1):
    """
    Solve the Quadrant-SVP using the Hill Climbing approach, constrained to the third quadrant.

    Parameters:
    - basis: The lattice basis (numpy array of shape (dimension, dimension)).
    - max_iterations: Maximum number of iterations for the algorithm.
    - step_size: Maximum step size for neighbor generation.

    Returns:
    - The shortest vector found that satisfies third quadrant constraints.
    - The input vector or lattice basis.
    """
    dimension = basis.shape[0]

    # Generate a random initial vector in one of the other quadrants
    initial_coefficients = np.random.randint(-10, 10, size=dimension)
    while not all(initial_coefficients < 0):  # Ensure it is not in the third quadrant
        initial_coefficients = np.random.randint(-10, 10, size=dimension)
    input_vector = np.dot(initial_coefficients, basis)

    # Start with a random vector in the third quadrant
    coefficients = -np.abs(np.random.randint(1, 10, size=dimension))
    current_vector = np.dot(coefficients, basis)
    current_value = objective_function(current_vector)

    for iteration in range(max_iterations):
        # Generate a neighbor by perturbing the coefficients within the third quadrant
        neighbor_coefficients = coefficients - np.abs(np.random.randint(0, step_size + 1, size=dimension))
        neighbor_vector = np.dot(neighbor_coefficients, basis)
        neighbor_value = objective_function(neighbor_vector)

        # If the neighbor is better, move to the neighbor
        if 0 < neighbor_value < current_value:  # Exclude zero vector
            coefficients = neighbor_coefficients
            current_vector = neighbor_vector
            current_value = neighbor_value

    return current_vector, current_value, input_vector

# Main execution block
def main():
    dimension = 2  # Dimension of the lattice
    size = 5      # Size of the basis vectors

    # Generate a random lattice
    lattice_basis = generate_random_lattice(dimension, size)

    # Measure computation time
    start_time = time.time()
    shortest_vector, shortest_value, input_vector = hill_climbing_quadrant_svp(lattice_basis)
    end_time = time.time()

    # Results
    print("Lattice Basis:")
    print(lattice_basis)
    print("\nInput Vector (Other Quadrant):", input_vector)
    print("Shortest Vector Found (Third Quadrant Constraint):", shortest_vector)
    print("Length of Shortest Vector (Third Quadrant Constraint):", shortest_value)
    print("Computation Time:", end_time - start_time, "seconds")

if __name__ == "__main__":
    main()
