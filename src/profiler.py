import jax
import jax.numpy as jnp
import time
import os
import csv
from monte_carlo import run_simulation

def profile_jax_performance():
    # JIT-compile the simulation function
    jitted_sim = jax.jit(run_simulation, static_argnums=(1,))
    
    sample_sizes = [10_000, 100_000, 1_000_000]
    results = []
    
    seed = 42
    key = jax.random.PRNGKey(seed)
    
    print(f"{'Samples':>12} | {'Cold Start (s)':>15} | {'Warm Start (s)':>15} | {'Speedup':>10}")
    print("-" * 60)
    
    for n in sample_sizes:
        # We need a new key each time
        key, subkey = jax.random.split(key)
        
        # Cold Start / JIT Compilation
        start = time.perf_counter()
        res_cold = jitted_sim(subkey, n).block_until_ready()
        cold_time = time.perf_counter() - start
        
        # Warm Start
        start = time.perf_counter()
        res_warm = jitted_sim(subkey, n).block_until_ready()
        warm_time = time.perf_counter() - start
        
        speedup = cold_time / warm_time if warm_time > 0 else 0
        
        results.append({
            "Samples": n,
            "Cold Start (s)": cold_time,
            "Warm Start (s)": warm_time,
            "Speedup": speedup
        })
        
        print(f"{n:12,d} | {cold_time:15.6f} | {warm_time:15.6f} | {speedup:10.2f}x")
        
    return results

if __name__ == "__main__":
    results = profile_jax_performance()
    
    # Save to CSV for report inclusion
    # Adjust output path to be relative to the script's location if needed, 
    # but the task says docs/Swarm_Stress_Report.md
    # docs is in genesis-oracle/docs
    output_dir = os.path.join("..", "docs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    csv_path = os.path.join(output_dir, "performance_metrics.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Samples", "Cold Start (s)", "Warm Start (s)", "Speedup"])
        writer.writeheader()
        writer.writerows(results)
    
    # Format for Swarm_Stress_Report.md
    report_md = "## JAX Performance Profiling Results\n\n"
    report_md += "| Sample Size | Cold Start (s) | Warm Start (s) | Speedup |\n"
    report_md += "|-------------|----------------|----------------|---------|\n"
    for row in results:
        report_md += f"| {int(row['Samples']):,} | {row['Cold Start (s)']: .6f} | {row['Warm Start (s)']: .6f} | {row['Speedup']: .2f}x |\n"
    
    report_path = os.path.join(output_dir, "Swarm_Stress_Report.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    
    print(f"\nResults saved to {csv_path} and {report_path}")
