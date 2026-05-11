import jax
import jax.numpy as jnp
from functools import partial

def oscillator_step(x, v, omega, dt=0.01, zeta=0.1):
    """
    Pure mathematical function for a single oscillator step.
    JAX requires pure functions.
    """
    # acceleration: a = -2*zeta*omega*v - omega^2*x
    accel = -2.0 * zeta * omega * v - (omega**2) * x
    
    # Update velocity and position (Euler Step)
    new_v = v + accel * dt
    new_x = x + new_v * dt
    
    return new_x, new_v

# Parallelize over the first dimension of x, v, and omega
vmapped_step = jax.vmap(oscillator_step, in_axes=(0, 0, 0, None, None))

@partial(jax.jit, static_argnames=['n_steps', 'dt', 'zeta'])
def run_simulation(x, v, omega, n_steps=1000, dt=0.01, zeta=0.1):
    """
    JIT-compiled simulation loop. 
    n_steps, dt, and zeta are marked static to allow JAX to optimize the loop.
    """
    def step_fn(state, _):
        x_curr, v_curr = state
        x_next, v_next = vmapped_step(x_curr, v_curr, omega, dt, zeta)
        return (x_next, v_next), None

    # Scan is the JAX-native way to do loops
    final_state, _ = jax.lax.scan(step_fn, (x, v), None, length=n_steps)
    return final_state
