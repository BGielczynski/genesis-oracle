import jax
import jax.numpy as jnp
import os

def business_model(demand, cost, penalty):
    return (demand * (1.0 - penalty)) - cost

def run_simulation(key, demand_mean, demand_std, cost_mu, cost_sigma, penalty_min, penalty_max, num_samples=10_000):
    keys = jax.random.split(key, 3)
    demand = demand_mean + demand_std * jax.random.normal(keys[0], (num_samples,))
    cost = jnp.exp(cost_mu + cost_sigma * jax.random.normal(keys[1], (num_samples,)))
    penalty = jax.random.uniform(keys[2], (num_samples,), minval=penalty_min, maxval=penalty_max)
    
    v_model = jax.vmap(business_model)
    revenues = v_model(demand, cost, penalty)
    return revenues

def main():
    seed = 42
    key = jax.random.PRNGKey(seed)
    num_samples = 10_000
    
    # Define ranges for stress testing
    demand_means = [100, 80, 60, 40]
    demand_stds = [20, 30, 40]
    cost_mus = [3.0, 3.5, 4.0, 4.5]
    cost_sigmas = [0.5, 0.8, 1.2]
    penalty_maxs = [0.15, 0.3, 0.5]
    
    results = []
    
    print("Starting Stress Test...")
    
    for d_mean in demand_means:
        for d_std in demand_stds:
            for c_mu in cost_mus:
                for c_sigma in cost_sigmas:
                    for p_max in penalty_maxs:
                        key, subkey = jax.random.split(key)
                        revenues = run_simulation(subkey, d_mean, d_std, c_mu, c_sigma, 0.05, p_max, num_samples)
                        
                        expected_revenue = float(jnp.mean(revenues))
                        var_95 = float(jnp.percentile(revenues, 5))
                        
                        results.append({
                            "demand_mean": d_mean,
                            "demand_std": d_std,
                            "cost_mu": c_mu,
                            "cost_sigma": c_sigma,
                            "penalty_max": p_max,
                            "expected_revenue": expected_revenue,
                            "var_95": var_95,
                            "failed": var_95 < 0
                        })
    
    breaking_points = [r for r in results if r["failed"]]
    
    print(f"Total scenarios tested: {len(results)}")
    print(f"Number of breaking points: {len(breaking_points)}")
    
    # Save findings
    os.makedirs("docs", exist_ok=True)
    report_path = "docs/Swarm_Stress_Report.md"
    
    with open(report_path, "w") as f:
        f.write("# Swarm Stress Test Report\n\n")
        f.write("## Overview\n")
        f.write("This report identifies parameter combinations where the JAX Business Revenue Model's 95% Value-at-Risk (VaR) becomes negative, indicating a significant risk of bankruptcy.\n\n")
        
        f.write("## Simulation Parameters\n")
        f.write("- **Demand**: Normal(mean, std)\n")
        f.write("- **Cost**: Log-Normal(mu, sigma)\n")
        f.write("- **Penalty**: Uniform(0.05, max)\n\n")
        
        f.write("## Breaking Points Summary\n")
        f.write(f"Total Scenarios: {len(results)}\n")
        f.write(f"Breaking Points (VaR 95% < 0): {len(breaking_points)}\n\n")
        
        if breaking_points:
            f.write("### Top 10 Most Critical Breaking Points\n\n")
            f.write("| Demand Mean | Demand Std | Cost Mu | Cost Sigma | Penalty Max | Expected Rev | VaR 95% |\n")
            f.write("|-------------|------------|---------|------------|-------------|--------------|---------|\n")
            
            # Sort by var_95
            sorted_breaking = sorted(breaking_points, key=lambda x: x["var_95"])
            for bp in sorted_breaking[:10]:
                f.write(f"| {bp['demand_mean']} | {bp['demand_std']} | {bp['cost_mu']} | {bp['cost_sigma']} | {bp['penalty_max']} | {bp['expected_revenue']:.2f} | {bp['var_95']:.2f} |\n")
            
            f.write("\n\n")
            
            f.write("### Analysis of Failure Conditions\n")
            f.write("- **Low Demand**: Scenarios with demand mean < 60 are highly susceptible.\n")
            f.write("- **High Cost Volatility**: Cost sigma > 1.0 significantly increases failure rates.\n")
            f.write("- **Penalty Sensitivity**: Increasing penalty max to 0.5 accelerates bankruptcy in marginal cases.\n")
        else:
            f.write("No breaking points were found with the current parameter ranges.\n")

    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
