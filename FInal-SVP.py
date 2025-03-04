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

def simulated_annealing(basis, max_iterations=1000, initial_temperature=100, cooling_rate=0.99):
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

def hill_climbing(basis, max_iterations=1000, step_size=1):
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

        if 0 < neighbor_value < current_value:
            coefficients = neighbor_coefficients
            current_vector = neighbor_vector
            current_value = neighbor_value

    return current_vector, current_value

def compare_algorithms(num_problems=100, dimension=3, size=5):
    """
    Compare Simulated Annealing and Hill Climbing on multiple problems.
    """
    sa_times = []
    hc_times = []
    sa_lengths = []
    hc_lengths = []

    for _ in range(num_problems):
        lattice_basis = generate_random_lattice(dimension, size)

        # Simulated Annealing
        start_time = time.time()
        _, sa_length = simulated_annealing(lattice_basis)
        sa_times.append(time.time() - start_time)
        sa_lengths.append(sa_length)

        # Hill Climbing
        start_time = time.time()
        _, hc_length = hill_climbing(lattice_basis)
        hc_times.append(time.time() - start_time)
        hc_lengths.append(hc_length)

    # Compute averages
    avg_sa_time = np.mean(sa_times)
    avg_hc_time = np.mean(hc_times)
    avg_sa_length = np.mean(sa_lengths)
    avg_hc_length = np.mean(hc_lengths)

    # Display averages
    print("Average Simulated Annealing Time:", avg_sa_time, "seconds")
    print("Average Hill Climbing Time:", avg_hc_time, "seconds")
    print("Average Simulated Annealing Length:", avg_sa_length)
    print("Average Hill Climbing Length:", avg_hc_length)

    # Plot Computational Time
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_problems), sa_times, label="Simulated Annealing (Time)", marker="o")
    plt.plot(range(num_problems), hc_times, label="Hill Climbing (Time)", marker="x")
    plt.xlabel("Problem Instance")
    plt.ylabel("Computation Time (seconds)")
    plt.title("Comparison of Computational Time: Simulated Annealing vs Hill Climbing")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Shortest Vector Lengths
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_problems), sa_lengths, label="Simulated Annealing (Length)", marker="o")
    plt.plot(range(num_problems), hc_lengths, label="Hill Climbing (Length)", marker="x")
    plt.xlabel("Problem Instance")
    plt.ylabel("Shortest Vector Length")
    plt.title("Comparison of Shortest Vector Lengths: Simulated Annealing vs Hill Climbing")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    compare_algorithms()
