import jax
import jax.numpy as jnp
from jax import random

def generate_pinn_data(key, n_colloc=5000, n_ic=500, n_bc=500):
    k1, k2, k3, k4, k5 = random.split(key, 5)
    
    # 1. Collocation Points (PDE)
    # Spatial domain x in [-1, 1], Temporal domain t in [0, 1]
    x_c = random.uniform(k1, shape=(n_colloc, 1), minval=-1.0, maxval=1.0)
    t_c = random.uniform(k2, shape=(n_colloc, 1), minval=0.0, maxval=1.0)
    colloc_pts = jnp.hstack((x_c, t_c))
    
    # 2. Initial Condition (IC) Points
    # 500 points at t = 0
    x_ic = random.uniform(k3, shape=(n_ic, 1), minval=-1.0, maxval=1.0)
    t_ic = jnp.zeros((n_ic, 1))
    ic_pts = jnp.hstack((x_ic, t_ic))
    # Starting temperature: u(x, 0) = -sin(pi * x)
    ic_vals = -jnp.sin(jnp.pi * x_ic)
    
    # 3. Boundary Condition (BC) Points
    # 500 points at the edges x = -1 and x = 1. We split this evenly.
    n_bc_half = n_bc // 2
    
    # Left boundary: x = -1
    t_bc_left = random.uniform(k4, shape=(n_bc_half, 1), minval=0.0, maxval=1.0)
    x_bc_left = -jnp.ones((n_bc_half, 1))
    bc_left_pts = jnp.hstack((x_bc_left, t_bc_left))
    # Dirichlet boundary condition: u(-1, t) = 0
    bc_left_vals = jnp.zeros((n_bc_half, 1))
    
    # Right boundary: x = 1
    t_bc_right = random.uniform(k5, shape=(n_bc - n_bc_half, 1), minval=0.0, maxval=1.0)
    x_bc_right = jnp.ones((n_bc - n_bc_half, 1))
    bc_right_pts = jnp.hstack((x_bc_right, t_bc_right))
    # Dirichlet boundary condition: u(1, t) = 0
    bc_right_vals = jnp.zeros((n_bc - n_bc_half, 1))
    
    # Combine boundary conditions
    bc_pts = jnp.vstack((bc_left_pts, bc_right_pts))
    bc_vals = jnp.vstack((bc_left_vals, bc_right_vals))
    
    return colloc_pts, ic_pts, ic_vals, bc_pts, bc_vals

if __name__ == "__main__":
    # Ensure deterministic chaos by managing PRNGKeys explicitly
    key = random.PRNGKey(42)
    colloc_pts, ic_pts, ic_vals, bc_pts, bc_vals = generate_pinn_data(key)
    
    print(f"Collocation points shape: {colloc_pts.shape}")
    print(f"Initial Condition points shape: {ic_pts.shape}")
    print(f"Initial Condition values shape: {ic_vals.shape}")
    print(f"Boundary Condition points shape: {bc_pts.shape}")
    print(f"Boundary Condition values shape: {bc_vals.shape}")
