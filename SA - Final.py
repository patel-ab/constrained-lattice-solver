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

def simulated_annealing_svp(basis, max_iterations=1000, initial_temperature=100, cooling_rate=0.99):
    """
    Solve the SVP using the Simulated Annealing approach.
    """
    dimension = basis.shape[0]
    coefficients = np.random.randint(-10, 10, size=dimension)
    current_vector = np.dot(coefficients, basis)
    current_value = objective_function(current_vector)
    best_vector = current_vector
    best_value = current_value
    temperature = initial_temperature

    for iteration in range(max_iterations):
        neighbor_coefficients = coefficients + np.random.randint(-1, 2, size=dimension)
        neighbor_vector = np.dot(neighbor_coefficients, basis)
        neighbor_value = objective_function(neighbor_vector)

        if np.all(neighbor_vector == 0):
            continue

        if neighbor_value < current_value or np.random.rand() < np.exp((current_value - neighbor_value) / temperature):
            coefficients = neighbor_coefficients
            current_vector = neighbor_vector
            current_value = neighbor_value

            if current_value < best_value:
                best_vector = current_vector
                best_value = current_value

        temperature *= cooling_rate

    return best_vector, best_value

def simulated_annealing_qsvp(basis, max_iterations=2000, initial_temperature=500, cooling_rate=0.85):
    """
    Solve the Quadrant-SVP using the Simulated Annealing approach, constrained to the third quadrant.
    """
    dimension = basis.shape[0]

    # Start with a random vector in the third quadrant
    coefficients = -np.abs(np.random.randint(1, 10, size=dimension))
    current_vector = np.dot(coefficients, basis)
    current_value = objective_function(current_vector)
    best_vector = current_vector
    best_value = current_value
    temperature = initial_temperature

    for _ in range(max_iterations):
        # Generate a neighbor by perturbing the coefficients within the third quadrant
        neighbor_coefficients = coefficients - np.abs(np.random.randint(0, 6, size=dimension))  # Increased range
        neighbor_vector = np.dot(neighbor_coefficients, basis)
        neighbor_value = objective_function(neighbor_vector)

        # Avoid zero vector
        if not any(neighbor_coefficients):
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

def compare_algorithms_svp_qsvp(num_problems=100, dimension=3, size=5):
    """
    Compare Simulated Annealing for SVP and QSVP.
    """
    svp_times = []
    qsvp_times = []
    svp_lengths = []
    qsvp_lengths = []

    for _ in range(num_problems):
        lattice_basis = generate_random_lattice(dimension, size)

        # Simulated Annealing for SVP
        start_time = time.time()
        svp_vector, svp_length = simulated_annealing_svp(lattice_basis)
        svp_times.append(time.time() - start_time)
        svp_lengths.append(svp_length)

        # Simulated Annealing for QSVP
        start_time = time.time()
        qsvp_vector, qsvp_length = simulated_annealing_qsvp(lattice_basis)
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
    plt.plot(range(num_problems), svp_times, label="Simulated Annealing SVP (Time)", marker="o")
    plt.plot(range(num_problems), qsvp_times, label="Simulated Annealing QSVP (Time)", marker="x")
    plt.xlabel("Problem Instance")
    plt.ylabel("Computation Time (seconds)")
    plt.title("Comparison of Computational Time: SVP vs QSVP")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Shortest Vector Lengths
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_problems), svp_lengths, label="Simulated Annealing SVP (Length)", marker="o")
    plt.plot(range(num_problems), qsvp_lengths, label="Simulated Annealing QSVP (Length)", marker="x")
    plt.xlabel("Problem Instance")
    plt.ylabel("Shortest Vector Length")
    plt.title("Comparison of Shortest Vector Lengths: SVP vs QSVP")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    compare_algorithms_svp_qsvp()
