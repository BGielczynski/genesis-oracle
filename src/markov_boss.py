import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os

def run_markov_simulation(key, num_days=365):
    # Transition Matrices
    # Row i, Column j: Probability of transition from state i to state j
    
    # Baseline Matrix
    P_base = jnp.array([
        [0.85, 0.12, 0.03], # Bull
        [0.10, 0.75, 0.15], # Stagnation
        [0.05, 0.20, 0.75]  # Recession
    ])
    
    # Shock Matrix (Day 180-190)
    # Mass from 0 and 1 shifted 80% to 2
    P_shock = jnp.array([
        [0.17, 0.024, 0.806],
        [0.02, 0.150, 0.830],
        [0.05, 0.200, 0.750]
    ])
    
    def step_fn(carry, x):
        key, current_state, day = carry
        
        # Determine which transition matrix to use
        # Shock period: 180 <= day <= 190
        is_shock = jnp.logical_and(day >= 180, day <= 190)
        P = jnp.where(is_shock, P_shock, P_base)
        
        # Get transition probabilities for current state
        probs = P[current_state]
        
        # Sample next state
        key, subkey = jax.random.split(key)
        next_state = jax.random.choice(subkey, 3, p=probs)
        
        return (key, next_state, day + 1), current_state

    # Initial state: Bull (0)
    initial_carry = (key, 0, 0)
    days = jnp.arange(num_days)
    
    _, state_history = jax.lax.scan(step_fn, initial_carry, days)
    
    return state_history

def main():
    seed = 123
    key = jax.random.PRNGKey(seed)
    
    num_days = 365
    print(f"Simulating Markov Economy for {num_days} days...")
    
    state_history = run_markov_simulation(key, num_days)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    days = jnp.arange(num_days)
    
    # Mapping states to names for the plot
    state_names = {0: "Bull", 1: "Stagnation", 2: "Recession"}
    
    plt.step(days, state_history, where='post', color='black', alpha=0.6)
    plt.fill_between(days, -0.5, 2.5, where=(days >= 180) & (days <= 190), 
                     color='red', alpha=0.2, label='Black Swan Shock')
    
    plt.yticks([0, 1, 2], ["Bull (0)", "Stagnation (1)", "Recession (2)"])
    plt.ylim(-0.5, 2.5)
    plt.xlabel("Day")
    plt.ylabel("Economic State")
    plt.title("Macro-Economic State Transitions over 1 Year")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    output_path = os.path.join("docs", "markov_states_history.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Count occurrences
    unique, counts = jnp.unique(state_history, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"State {state_names[int(u)]}: {c} days ({(c/num_days)*100:.1f}%)")

if __name__ == "__main__":
    main()
