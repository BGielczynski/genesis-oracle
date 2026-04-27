import os
os.environ["KERAS_BACKEND"] = "jax"

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def generate_square_wave_fourier(student_id_last_digits=89, num_harmonics=9, num_periods=100):
    """
    Generates a Fourier series approximation of a square wave.
    
    Parameters:
    - student_id_last_digits: Used as the period T.
    - num_harmonics: Number of odd harmonics to include (e.g., 9 means up to the 17th harmonic).
    - num_periods: Total number of periods to generate.
    """
    T = student_id_last_digits
    omega_0 = 2 * jnp.pi / T
    
    # Time vector: 100 periods, with 1000 points per period for smoothness
    t = jnp.linspace(0, num_periods * T, num_periods * 1000)
    
    # Fourier series summation
    # f(t) = (4/pi) * sum_{k=1,3,5,...}^N (1/k) * sin(k * omega_0 * t)
    f_t = jnp.zeros_like(t)
    
    harmonics_indices = [2*i + 1 for i in range(num_harmonics)]
    
    for k in harmonics_indices:
        f_t += (1.0 / k) * jnp.sin(k * omega_0 * t)
    
    f_t = (4.0 / jnp.pi) * f_t
    
    return t, f_t, T, omega_0, harmonics_indices

def apply_rc_filter(t, T, omega_0, harmonics_indices, R=500, C=1189e-6):
    """
    Applies an RC low-pass filter to the Fourier series.
    
    H(omega) = 1 / (1 + j*omega*R*C)
    """
    RC = R * C
    f_t_filtered = jnp.zeros_like(t)
    
    print(f"Applying RC filter: R={R} Ohm, C={C*1e6} uF (RC={RC:.4f}s)")
    
    for k in harmonics_indices:
        omega = k * omega_0
        # |H(omega)| = 1 / sqrt(1 + (omega*RC)^2)
        amplitude_gain = 1.0 / jnp.sqrt(1 + (omega * RC)**2)
        # phase = -arctan(omega*RC)
        phase_shift = -jnp.arctan(omega * RC)
        
        # Original component: (1/k) * sin(omega * t)
        # Filtered: (1/k) * |H| * sin(omega * t + phase)
        f_t_filtered += (1.0 / k) * amplitude_gain * jnp.sin(omega * t + phase_shift)
        
        print(f"Harmonic {k}: Gain={amplitude_gain:.4f}, Phase={jnp.degrees(phase_shift):.2f} deg")

    f_t_filtered = (4.0 / jnp.pi) * f_t_filtered
    return f_t_filtered

def plot_comparison(t, f_t, f_t_filtered, T, num_show_periods=5):
    """Plots original vs filtered signal."""
    plt.figure(figsize=(12, 6))
    mask = t <= num_show_periods * T
    
    plt.plot(t[mask], f_t[mask], label="Original Square Wave (9 harmonics)", alpha=0.6, linestyle='--')
    plt.plot(t[mask], f_t_filtered[mask], label="RC Filtered Signal", linewidth=2)
    
    plt.title(f"Effect of RC Low-Pass Filter (T={T}, R=500 Ohm, C=1189 uF)")
    plt.xlabel("Time (t)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/fourier_rc_filter_comparison.png")
    print("Comparison plot saved to docs/fourier_rc_filter_comparison.png")
    plt.close()

def add_noise_and_spikes(t, f_t, T, noise_std=0.05):
    """Adds Gaussian noise and injects a massive spike between periods 70 and 75."""
    import jax
    key = jax.random.PRNGKey(42)
    
    # Add Gaussian noise
    noise = jax.random.normal(key, f_t.shape) * noise_std
    f_t_corrupted = f_t + noise
    
    # Inject massive high-frequency spike between period 70 and 75
    # Mask for the time range [70*T, 75*T]
    spike_mask = (t >= 70 * T) & (t <= 75 * T)
    
    # High frequency spike: large amplitude sine wave at much higher frequency
    spike_freq = 50 * (2 * jnp.pi / T)  # 50x fundamental
    spike_amplitude = 5.0  # Massive compared to base signal (~1.0)
    
    spike = spike_amplitude * jnp.sin(spike_freq * t)
    
    # Apply spike only where the mask is True
    f_t_corrupted = jnp.where(spike_mask, f_t_corrupted + spike, f_t_corrupted)
    
    print(f"Injected spikes between t={70*T:.2f} and t={75*T:.2f}")
    return f_t_corrupted

def save_data(f_t, filename="data/corrupted_signal.npy"):
    """Saves the 1D signal to the data folder."""
    os.makedirs("data", exist_ok=True)
    np.save(filename, f_t)
    print(f"Signal saved to {filename}")

def plot_data_feed(t, f_t_final, T):
    """Plots a window of normal signal and a window with the anomaly."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Normal window (e.g., periods 10-15)
    mask_normal = (t >= 10 * T) & (t <= 15 * T)
    ax1.plot(t[mask_normal], f_t_final[mask_normal], color='blue')
    ax1.set_title("Normal Noisy Signal (Periods 10-15)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    
    # Anomaly window (periods 68-77 to see the spike context)
    mask_anomaly = (t >= 68 * T) & (t <= 77 * T)
    ax2.plot(t[mask_anomaly], f_t_final[mask_anomaly], color='red')
    ax2.set_title("Signal with Anomaly Spike (Periods 70-75)")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("docs/data_feed.png")
    print("Data feed plot saved to docs/data_feed.png")
    plt.close()

if __name__ == "__main__":
    # Parameters from student ID
    ID_LAST_DIGITS = 89
    ID_LAST_THREE = 189
    R = 500  # 0.5 kOhm
    C = (1000 + ID_LAST_THREE) * 1e-6  # 1189 uF
    
    t, f_t, T, omega_0, harmonics = generate_square_wave_fourier(student_id_last_digits=ID_LAST_DIGITS)
    
    f_t_filtered = apply_rc_filter(t, T, omega_0, harmonics, R=R, C=C)
    
    # Add noise and corruption
    f_t_final = add_noise_and_spikes(t, f_t_filtered, T)
    
    # Save the data
    save_data(f_t_final)
    
    # Generate requested plots
    print(f"Generated signals with {len(t)} points.")
    plot_comparison(t, f_t, f_t_final, T)
    plot_data_feed(t, f_t_final, T)
