import numpy as np
import time

def generate_random_lattice(dimension, size):
    """
    Generate a random lattice with the given dimension and size.
    """
    basis = np.random.randint(-size, size, size=(dimension, dimension))
    return basis

def objective_function(vector):
    """
    Objective function to minimize: the Euclidean norm of the vector.
    """
    return np.linalg.norm(vector)

def hill_climbing(basis, max_iterations=1000, step_size=1):
    """
    Solve the SVP using the Hill Climbing approach.

    Parameters:
    - basis: The lattice basis (numpy array of shape (dimension, dimension)).
    - max_iterations: Maximum number of iterations for the algorithm.
    - step_size: Maximum step size for neighbor generation.

    Returns:
    - The shortest vector found.
    """
    dimension = basis.shape[0]

    # Generate a random initial vector in the lattice
    coefficients = np.random.randint(-10, 10, size=dimension)
    current_vector = np.dot(coefficients, basis)
    current_value = objective_function(current_vector)

    for iteration in range(max_iterations):
        # Generate a neighbor by perturbing the coefficients
        neighbor_coefficients = coefficients + np.random.randint(-step_size, step_size + 1, size=dimension)
        neighbor_vector = np.dot(neighbor_coefficients, basis)
        neighbor_value = objective_function(neighbor_vector)

        # If the neighbor is better, move to the neighbor
        if 0 < neighbor_value < current_value:  # Exclude zero vector
            coefficients = neighbor_coefficients
            current_vector = neighbor_vector
            current_value = neighbor_value

    return current_vector, current_value

# Main execution block
def main():
    dimension = 2  # Dimension of the lattice
    size = 5      # Size of the basis vectors

    # Generate a random lattice
    lattice_basis = generate_random_lattice(dimension, size)

    # Measure computation time
    start_time = time.time()
    shortest_vector, shortest_value = hill_climbing(lattice_basis)
    end_time = time.time()

    # Results
    print("Lattice Basis:")
    print(lattice_basis)
    print("\nShortest Vector Found:", shortest_vector)
    print("Length of Shortest Vector:", shortest_value)
    print("Computation Time:", end_time - start_time, "seconds")

if __name__ == "__main__":
    main()