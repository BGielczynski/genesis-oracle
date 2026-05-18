import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from jax import random

# ==========================================
# EXERCISE 2: Agentic Weaving (The Neural Surrogate in Flax)
# ==========================================

class HeatSurrogate(nn.Module):
    """
    A Neural Surrogate mapping continuous 2D spacetime (x, t) to a 1D temperature field (u).
    Using 4 hidden layers with 32 neurons each and tanh activations for smooth second derivatives.
    """
    @nn.compact
    def __call__(self, x, t):
        # Stack space and time inputs into a single feature vector
        inputs = jnp.hstack([x, t])
        
        # 4 Hidden Layers
        h = nn.Dense(32)(inputs)
        h = nn.tanh(h)
        h = nn.Dense(32)(h)
        h = nn.tanh(h)
        h = nn.Dense(32)(h)
        h = nn.tanh(h)
        h = nn.Dense(32)(h)
        h = nn.tanh(h)
        
        # Output layer for 1D scalar temperature
        u = nn.Dense(1)(h)
        return u

# Wrapper function for the model so it can be easily differentiated
def predict_u(params, x, t):
    model = HeatSurrogate()
    # The model returns an array of shape (1,), extract the scalar at index 0
    return model.apply(params, x, t)[0]

# ==========================================
# EXERCISE 3: The Differentiable Fabric (jax.grad & The Physics Loss)
# ==========================================

# Thermal diffusivity parameter
alpha = 0.01

# Calculate analytical derivatives using jax.grad
# predict_u signature: params (argnum=0), x (argnum=1), t (argnum=2)
u_t_fn = jax.grad(predict_u, argnums=2)       # First derivative w.r.t time t
u_x_fn = jax.grad(predict_u, argnums=1)       # First derivative w.r.t space x
u_xx_fn = jax.grad(u_x_fn, argnums=1)         # Second derivative w.r.t space x

def pde_residual(params, x, t):
    """
    Calculates the 1D Heat Equation residual: u_t - alpha * u_xx = 0
    """
    u_t = u_t_fn(params, x, t)
    u_xx = u_xx_fn(params, x, t)
    return u_t - alpha * u_xx

# Vectorize the residual and prediction functions over batches of (x, t) points
# in_axes=(None, 0, 0) means params are identical, but we map over the first dimension of x and t arrays
vmap_pde_residual = jax.vmap(pde_residual, in_axes=(None, 0, 0))
vmap_predict_u = jax.vmap(predict_u, in_axes=(None, 0, 0))

def compute_loss(params, colloc_pts, ic_pts, ic_vals, bc_pts, bc_vals):
    """
    Computes the total loss combining physics, initial condition, and boundary conditions.
    """
    # 1. Physics Loss (Collocation Points)
    x_c, t_c = colloc_pts[:, 0], colloc_pts[:, 1]
    pde_res = vmap_pde_residual(params, x_c, t_c)
    loss_physics = jnp.mean(pde_res**2)
    
    # 2. Initial Condition (IC) Loss
    x_ic, t_ic = ic_pts[:, 0], ic_pts[:, 1]
    u_ic_pred = vmap_predict_u(params, x_ic, t_ic)
    u_ic_pred = u_ic_pred.reshape(-1, 1) # Ensure shape matches ic_vals (N, 1)
    loss_ic = jnp.mean((u_ic_pred - ic_vals)**2)
    
    # 3. Boundary Condition (BC) Loss
    x_bc, t_bc = bc_pts[:, 0], bc_pts[:, 1]
    u_bc_pred = vmap_predict_u(params, x_bc, t_bc)
    u_bc_pred = u_bc_pred.reshape(-1, 1)
    loss_bc = jnp.mean((u_bc_pred - bc_vals)**2)
    
    # Total Loss aggregation
    total_loss = loss_physics + loss_ic + loss_bc
    return total_loss

if __name__ == "__main__":
    print("Initializing Physics-Informed Neural Network (PINN)...")
    
    # Explicitly initialize the model's weights using jax.random.PRNGKey
    key = random.PRNGKey(42)
    model = HeatSurrogate()
    
    # Create dummy inputs to initialize shapes (1 sample, 1 feature)
    dummy_x = jnp.zeros(())
    dummy_t = jnp.zeros(())
    
    params = model.init(key, dummy_x, dummy_t)
    print("Neural Surrogate Architecture initialized successfully.")
    
    # Basic Optax training loop setup
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    print("Optax Adam Optimizer initialized successfully.")
    
    # Cloud Ignition training (Exercise 4) to follow in Colab...
