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

def compare_algorithms_qsvp(num_problems=100, dimension=3, size=5):
    """
    Compare Hill Climbing and Simulated Annealing for Quadrant-SVP.
    """
    hc_times = []
    sa_times = []
    hc_lengths = []
    sa_lengths = []

    for _ in range(num_problems):
        lattice_basis = generate_random_lattice(dimension, size)

        # Hill Climbing
        start_time = time.time()
        hc_vector, hc_length = hill_climbing_qsvp(lattice_basis)
        hc_times.append(time.time() - start_time)
        hc_lengths.append(hc_length)

        # Simulated Annealing
        start_time = time.time()
        sa_vector, sa_length = simulated_annealing_qsvp(lattice_basis)
        sa_times.append(time.time() - start_time)
        sa_lengths.append(sa_length)

    # Compute averages
    avg_hc_time = np.mean(hc_times)
    avg_sa_time = np.mean(sa_times)
    avg_hc_length = np.mean(hc_lengths)
    avg_sa_length = np.mean(sa_lengths)

    # Display averages
    print("Average Hill Climbing Time:", avg_hc_time, "seconds")
    print("Average Simulated Annealing Time:", avg_sa_time, "seconds")
    print("Average Hill Climbing Length:", avg_hc_length)
    print("Average Simulated Annealing Length:", avg_sa_length)

    # Plot Computational Time
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_problems), hc_times, label="Hill Climbing (Time)", marker="o")
    plt.plot(range(num_problems), sa_times, label="Simulated Annealing (Time)", marker="x")
    plt.xlabel("Problem Instance")
    plt.ylabel("Computation Time (seconds)")
    plt.title("Comparison of Computational Time: Hill Climbing vs Simulated Annealing")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Shortest Vector Lengths
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_problems), hc_lengths, label="Hill Climbing (Length)", marker="o")
    plt.plot(range(num_problems), sa_lengths, label="Simulated Annealing (Length)", marker="x")
    plt.xlabel("Problem Instance")
    plt.ylabel("Shortest Vector Length")
    plt.title("Comparison of Shortest Vector Lengths: Hill Climbing vs Simulated Annealing")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    compare_algorithms_qsvp()
