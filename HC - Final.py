import numpy as np
import time
import matplotlib.pyplot as plt

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

def hill_climbing_svp(basis, max_iterations=1000, step_size=1):
    """
    Solve the SVP using the Hill Climbing approach.
    """
    dimension = basis.shape[0]
    coefficients = np.random.randint(-10, 10, size=dimension)
    current_vector = np.dot(coefficients, basis)
    current_value = objective_function(current_vector)

    for iteration in range(max_iterations):
        neighbor_coefficients = coefficients + np.random.randint(-step_size, step_size + 1, size=dimension)
        neighbor_vector = np.dot(neighbor_coefficients, basis)
        neighbor_value = objective_function(neighbor_vector)

        if 0 < neighbor_value < current_value:  # Exclude zero vector
            coefficients = neighbor_coefficients
            current_vector = neighbor_vector
            current_value = neighbor_value

    return current_vector, current_value

def hill_climbing_qsvp(basis, max_iterations=1000, step_size=1):
    """
    Solve the Quadrant-SVP using the Hill Climbing approach, constrained to the third quadrant.
    """
    dimension = basis.shape[0]

    # Start with a random vector in the third quadrant
    coefficients = -np.abs(np.random.randint(1, 10, size=dimension))
    current_vector = np.dot(coefficients, basis)
    current_value = objective_function(current_vector)

    for _ in range(max_iterations):
        # Generate a neighbor by perturbing the coefficients within the third quadrant
        neighbor_coefficients = coefficients - np.abs(np.random.randint(0, step_size + 1, size=dimension))
        neighbor_vector = np.dot(neighbor_coefficients, basis)
        neighbor_value = objective_function(neighbor_vector)

        # If the neighbor is better, move to the neighbor
        if 0 < neighbor_value < current_value:  # Exclude zero vector
            coefficients = neighbor_coefficients
            current_vector = neighbor_vector
            current_value = neighbor_value

    return current_vector, current_value

def compare_algorithms_svp_qsvp(num_problems=100, dimension=3, size=5):
    """
    Compare Hill Climbing for SVP and QSVP.
    """
    svp_times = []
    qsvp_times = []
    svp_lengths = []
    qsvp_lengths = []

    for _ in range(num_problems):
        lattice_basis = generate_random_lattice(dimension, size)

        # Hill Climbing for SVP
        start_time = time.time()
        svp_vector, svp_length = hill_climbing_svp(lattice_basis)
        svp_times.append(time.time() - start_time)
        svp_lengths.append(svp_length)

        # Hill Climbing for QSVP
        start_time = time.time()
        qsvp_vector, qsvp_length = hill_climbing_qsvp(lattice_basis)
        qsvp_times.append(time.time() - start_time)
        qsvp_lengths.append(qsvp_length)

    # Compute averages
    avg_svp_time = np.mean(svp_times)
    avg_qsvp_time = np.mean(qsvp_times)
    avg_svp_length = np.mean(svp_lengths)
    avg_qsvp_length = np.mean(qsvp_lengths)

    # Display averages
    print("Average SVP Time:", avg_svp_time, "seconds")
    print("Average QSVP Time:", avg_qsvp_time, "seconds")
    print("Average SVP Length:", avg_svp_length)
    print("Average QSVP Length:", avg_qsvp_length)

    # Plot Computational Time
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_problems), svp_times, label="Hill Climbing SVP (Time)", marker="o")
    plt.plot(range(num_problems), qsvp_times, label="Hill Climbing QSVP (Time)", marker="x")
    plt.xlabel("Problem Instance")
    plt.ylabel("Computation Time (seconds)")
    plt.title("Comparison of Computational Time: SVP vs QSVP")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Shortest Vector Lengths
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_problems), svp_lengths, label="Hill Climbing SVP (Length)", marker="o")
    plt.plot(range(num_problems), qsvp_lengths, label="Hill Climbing QSVP (Length)", marker="x")
    plt.xlabel("Problem Instance")
    plt.ylabel("Shortest Vector Length")
    plt.title("Comparison of Shortest Vector Lengths: SVP vs QSVP")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    compare_algorithms_svp_qsvp()
