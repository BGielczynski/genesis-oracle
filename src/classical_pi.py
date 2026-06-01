import numpy as np
import matplotlib.pyplot as plt
import time
import os

def estimate_pi(num_points):
    start_time = time.time()
    
    # Generate random points
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    
    # Calculate distance from origin
    distance = x**2 + y**2
    
    # Points inside the circle
    inside_circle = distance <= 1
    num_inside = np.sum(inside_circle)
    
    # Estimate Pi
    pi_estimate = 4 * num_inside / num_points
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return pi_estimate, execution_time, x, y, inside_circle

def main():
    num_points = 5_000_000
    print(f"Estimating Pi with {num_points:,} points...")
    
    pi_est, exec_time, x, y, inside = estimate_pi(num_points)
    
    print(f"Estimated Pi: {pi_est}")
    print(f"Execution Time: {exec_time:.4f} seconds")
    
    # Visualization (subset for performance)
    plot_points = 10_000
    plt.figure(figsize=(8, 8))
    plt.scatter(x[:plot_points][inside[:plot_points]], y[:plot_points][inside[:plot_points]], color='blue', s=1, label='Inside')
    plt.scatter(x[:plot_points][~inside[:plot_points]], y[:plot_points][~inside[:plot_points]], color='red', s=1, label='Outside')
    
    circle = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_artist(circle)
    
    plt.axis('equal')
    plt.title(f"Monte Carlo Pi Estimation (N={num_points})\nEstimate: {pi_est:.6f}, Time: {exec_time:.4f}s")
    plt.legend()
    
    output_path = os.path.join("docs", "classical_pi_disp.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
