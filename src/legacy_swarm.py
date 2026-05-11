import numpy as np
import time

def simulate_swarm():
    """
    Simulates 100,000 damped harmonic oscillators using explicit Euler integration.
    """
    n_pendulums = 100000
    n_steps = 1000
    dt = 0.01
    zeta = 0.1  # Damping ratio
    
    # Initialize 100,000 random natural frequencies
    omega_0 = np.random.uniform(1.0, 5.0, n_pendulums)
    
    # Initialize state: positions at 1.0, velocities at 0.0
    x = np.ones(n_pendulums, dtype=np.float64)
    v = np.zeros(n_pendulums, dtype=np.float64)
    
    # Record start time
    start_time = time.time()
    
    # Simulation loop (Python for-loop over time steps)
    for _ in range(n_steps):
        # acceleration: a = -2*zeta*omega_0*v - omega_0^2*x
        accel = -2.0 * zeta * omega_0 * v - (omega_0**2) * x
        
        # Update velocity and position (Euler Step)
        v += accel * dt
        x += v * dt
        
    # Record end time
    end_time = time.time()
    
    return end_time - start_time

if __name__ == "__main__":
    print("Starting simulation of 100,000 pendulums...")
    total_time = simulate_swarm()
    print(f"Total execution time: {total_time:.4f} seconds")
