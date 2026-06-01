import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
import os

def business_model(demand, cost, penalty):
    """
    Simple business revenue model.
    Revenue = (Demand * (1 - Penalty)) - Cost
    """
    return (demand * (1.0 - penalty)) - cost

def run_simulation(key, num_samples=10_000):
    keys = jax.random.split(key, 3)
    
    # 1. Market Demand: Normal(100, 20)
    demand = 100 + 20 * jax.random.normal(keys[0], (num_samples,))
    
    # 2. Asset Cost: Log-Normal(3, 0.5)
    # Log-Normal is exp(Normal(mu, sigma))
    cost = jnp.exp(3.0 + 0.5 * jax.random.normal(keys[1], (num_samples,)))
    
    # 3. Penalty Rate: Uniform(0.05, 0.15)
    penalty = jax.random.uniform(keys[2], (num_samples,), minval=0.05, maxval=0.15)
    
    # Vectorized execution using vmap
    v_model = jax.vmap(business_model)
    revenues = v_model(demand, cost, penalty)
    
    return revenues

def main():
    seed = 42
    key = jax.random.PRNGKey(seed)
    
    num_samples = 10_000
    print(f"Running JAX Monte Carlo with {num_samples:,} samples...")
    
    start_time = time.time()
    revenues = run_simulation(key, num_samples)
    # Trigger JIT by calculating something
    mean_rev = jnp.mean(revenues)
    exec_time = time.time() - start_time
    
    # Statistics
    expected_revenue = jnp.mean(revenues)
    var_95 = jnp.percentile(revenues, 5) # 5th percentile for 95% VaR
    
    print(f"Expected Revenue: {expected_revenue:.2f}")
    print(f"Value-at-Risk (95%): {var_95:.2f}")
    print(f"Execution Time: {exec_time:.4f} seconds")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(revenues, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(expected_revenue, color='blue', linestyle='dashed', linewidth=2, label=f'Expected: {expected_revenue:.2f}')
    plt.axvline(var_95, color='red', linestyle='dashed', linewidth=2, label=f'VaR 95%: {var_95:.2f}')
    
    plt.title(f"JAX Monte Carlo Business Simulation (N={num_samples})")
    plt.xlabel("Revenue")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    output_path = os.path.join("docs", "monte_carlo_revenue.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
