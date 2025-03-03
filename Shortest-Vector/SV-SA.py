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

def simulated_annealing(basis, max_iterations=1000, initial_temperature=100, cooling_rate=0.99):
    """
    Solve the SVP using the Simulated Annealing approach.

    Parameters:
    - basis: The lattice basis (numpy array of shape (dimension, dimension)).
    - max_iterations: Maximum number of iterations for the algorithm.
    - initial_temperature: Starting temperature for the annealing process.
    - cooling_rate: Rate at which the temperature decreases.

    Returns:
    - The shortest vector found.
    """
    dimension = basis.shape[0]

    # Generate a random initial vector in the lattice
    coefficients = np.random.randint(-10, 10, size=dimension)
    current_vector = np.dot(coefficients, basis)
    current_value = objective_function(current_vector)

    best_vector = current_vector
    best_value = current_value

    temperature = initial_temperature

    for iteration in range(max_iterations):
        # Generate a neighbor by perturbing the coefficients
        neighbor_coefficients = coefficients + np.random.randint(-1, 2, size=dimension)
        neighbor_vector = np.dot(neighbor_coefficients, basis)
        neighbor_value = objective_function(neighbor_vector)

        # Avoid zero vector as a solution
        if np.all(neighbor_vector == 0):
            continue

        # Decide whether to accept the neighbor
        if neighbor_value < current_value or np.random.rand() < np.exp((current_value - neighbor_value) / temperature):
            coefficients = neighbor_coefficients
            current_vector = neighbor_vector
            current_value = neighbor_value

            # Update the best solution if the neighbor is better
            if current_value < best_value:
                best_vector = current_vector
                best_value = current_value

        # Cool down the temperature
        temperature *= cooling_rate

    return best_vector, best_value

# Main execution block
def main():
    dimension = 2  # Dimension of the lattice
    size = 5      # Size of the basis vectors

    # Generate a random lattice
    lattice_basis = generate_random_lattice(dimension, size)

    # Measure computation time
    start_time = time.time()
    shortest_vector, shortest_value = simulated_annealing(lattice_basis)
    end_time = time.time()

    # Results
    print("Lattice Basis:")
    print(lattice_basis)
    print("\nShortest Vector Found:", shortest_vector)
    print("Length of Shortest Vector:", shortest_value)
    print("Computation Time:", end_time - start_time, "seconds")

if __name__ == "__main__":
    main()
