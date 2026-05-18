# ==========================================
# CELL 1: Initialization & Environment Setup
# ==========================================
# Run this cell directly in Colab
!git clone https://github.com/BGielczynski/genesis-oracle.git
%cd genesis-oracle
!pip install -q jax jaxlib flax optax plotly tqdm

import jax
import jax.numpy as jnp
from jax import random
import optax
import plotly.graph_objects as go
from tqdm.auto import tqdm

# Import the data generation and the model/loss functions
from src.pinn_data import generate_pinn_data
from src.fabric_pinn import HeatSurrogate, compute_loss, vmap_predict_u

# ==========================================
# CELL 2: Silicon Ignition (Training the PINN)
# ==========================================
key = random.PRNGKey(42)
key, subkey = random.split(key)

# 1. Generate Data
colloc_pts, ic_pts, ic_vals, bc_pts, bc_vals = generate_pinn_data(subkey)

# 2. Initialize Model and Optimizer
model = HeatSurrogate()
params = model.init(key, jnp.zeros((1,)), jnp.zeros((1,)))
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

# 3. Define JIT-compiled training step
@jax.jit
def train_step(params, opt_state, colloc_pts, ic_pts, ic_vals, bc_pts, bc_vals):
    loss, grads = jax.value_and_grad(compute_loss)(params, colloc_pts, ic_pts, ic_vals, bc_pts, bc_vals)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# 4. Training Loop
epochs = 5000
print("Starting Cloud Ignition (Training)...")
for epoch in tqdm(range(epochs)):
    params, opt_state, loss = train_step(params, opt_state, colloc_pts, ic_pts, ic_vals, bc_pts, bc_vals)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

print(f"Final Loss at Epoch {epochs}: {loss:.6f}")

# ==========================================
# CELL 3: The Interactive 3D Projection
# ==========================================
# Create a high-resolution meshgrid for inference
x_linspace = jnp.linspace(-1.0, 1.0, 100)
t_linspace = jnp.linspace(0.0, 1.0, 100)
X, T = jnp.meshgrid(x_linspace, t_linspace)

# Flatten meshgrid to pass through the model
x_flat = X.flatten()
t_flat = T.flatten()

# Predict continuous temperature field
u_pred_flat = vmap_predict_u(params, x_flat, t_flat)
U_pred = u_pred_flat.reshape(X.shape)

# Create an interactive 3D Surface Plot
fig = go.Figure(data=[go.Surface(z=U_pred, x=X, y=T, colorscale='inferno')])
fig.update_layout(
    title='PINN Solution to the 1D Heat Equation',
    scene=dict(
        xaxis_title='Space (x)',
        yaxis_title='Time (t)',
        zaxis_title='Temperature (u)'
    ),
    width=800, height=800
)

# Display in Colab
fig.show()

# Save as HTML (Ensure the docs directory exists)
import os
os.makedirs("docs", exist_ok=True)
fig.write_html("docs/pinn_3d_fabric.html")
print("Saved interactive plot to docs/pinn_3d_fabric.html")
