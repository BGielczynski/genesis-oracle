# 🚀 Colab Training Plan: PhysicsAutoencoder (TPU / GPU / CPU)

This document provides a step-by-step guide and the corresponding code cells to train and evaluate the PhysicsAutoencoder using Google Colab. Support for TPU, GPU (T4), and CPU is included.

---

## 🛠 Step 1: Environment & Hardware Setup
This cell configures the Keras backend, handles repository paths, and verifies the hardware (TPU, GPU, or CPU).

```python
import os
import sys

# 1. Set Keras backend to JAX
os.environ["KERAS_BACKEND"] = "jax"

# 2. Hardware Selection (Uncomment to force a specific platform, otherwise JAX auto-detects)
# os.environ["JAX_PLATFORMS"] = "tpu"   # Use for TPU v5e / v2
# os.environ["JAX_PLATFORMS"] = "cuda"  # Use for GPU T4 / L4 / A100
# os.environ["JAX_PLATFORMS"] = "cpu"   # Use for CPU-only mode

# 3. Repository Setup
repo_url = "https://github.com/BGielczynski/genesis-oracle"
repo_name = "genesis-oracle"

if not os.path.exists(repo_name):
    !git clone {repo_url}

%cd {repo_name}
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import jax
import keras

# 4. Verify Hardware
devices = jax.devices()
print(f"JAX Devices: {devices}")
device_type = devices[0].device_kind.lower()

if "tpu" in device_type:
    print(f"✅ Success! Running on TPU: {devices[0].device_kind}")
elif "gpu" in device_type or "cuda" in device_type:
    print(f"🚀 Success! Running on GPU (T4): {devices[0].device_kind}")
else:
    print(f"💻 Note: Running on CPU: {devices[0].device_kind}")
```

---

## 📦 Step 2: Imports & Model Building
Load the custom layers and the PhysicsAutoencoder from the source files.

```python
from src.data_generator import generate_square_wave_fourier, apply_rc_filter, add_noise_and_spikes, save_data
from src.architecture import PhysicsAutoencoder, prepare_datasets

# Instantiate and compile the model
# Using window_size=50 and latent_dim=8
autoencoder = PhysicsAutoencoder(window_size=50, latent_dim=8)
autoencoder.compile(optimizer="adam", loss="mse")
print("Model initialized and compiled for TPU.")
```

---

## 📊 Step 3: Physical Data Generation
Generate the noisy signal with the RC filter and prepare overlapping windows for training.

```python
print("Generating signal data...")
# Physics Parameters (Student-ID: 89, Last 3: 189)
ID_LAST_DIGITS = 89
ID_LAST_THREE = 189
R = 500
C = (1000 + ID_LAST_THREE) * 1e-6

# 1. Fourier Approximation -> RC Filter -> Noise & Spikes
t, f_t, T, omega_0, harmonics = generate_square_wave_fourier(student_id_last_digits=ID_LAST_DIGITS)
f_t_filtered = apply_rc_filter(t, T, omega_0, harmonics, R=R, C=C)
f_t_final = add_noise_and_spikes(t, f_t_filtered, T)

# 2. Save and Load as Sliced Datasets
save_data(f_t_final, filename="data/corrupted_signal.npy")
X_train, X_test = prepare_datasets("data/corrupted_signal.npy")

# Convert to float32 for TPU efficiency
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

print(f"Datasets ready. Training windows: {X_train.shape[0]}")
```

---

## 🔥 Step 4: Model Training (30 Epochs)
Train the autoencoder on normal data. Watch the JAX XLA compiler speed boost after Epoch 1.

```python
print("Starting TPU-accelerated training...")
# Larger batch sizes (256+) are more efficient on TPUs
history = autoencoder.fit(
    X_train, 
    X_train, 
    epochs=30, 
    batch_size=256, 
    shuffle=True,
    verbose=1
)
print("Training complete!")
```

---

## 📈 Step 5: Loss Visualization
Monitor the Mean Squared Error (MSE) to ensure convergence.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss', color='#3498db', linewidth=2)
plt.title('PhysicsAutoencoder Training Progress (TPU)', fontsize=14)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (Log Scale)')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

---

## 🚨 Step 6: Anomaly Detection Test
Compare the reconstruction error on normal data vs. the data containing the anomaly spike.

```python
import numpy as np

# 1. Predict reconstructions
reconstructions_train = autoencoder.predict(X_train, batch_size=256)
reconstructions_test = autoencoder.predict(X_test, batch_size=256)

# 2. Calculate MAE per window
mae_train = np.mean(np.abs(X_train - reconstructions_train), axis=1)
mae_test = np.mean(np.abs(X_test - reconstructions_test), axis=1)

# 3. Plot Comparison
plt.figure(figsize=(12, 6))
plt.plot(mae_train, label='Reconstruction Error (Normal)', alpha=0.7)
plt.plot(range(len(mae_train), len(mae_train) + len(mae_test)), mae_test, label='Reconstruction Error (Anomaly)', color='red', alpha=0.7)
plt.axhline(y=np.max(mae_train) * 1.5, color='black', linestyle='--', label='Anomaly Threshold')
plt.title('Anomaly Detection: MAE Reconstruction Error Comparison')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Window Index')
plt.legend()
plt.show()

print(f"Max Training MAE: {np.max(mae_train):.6f}")
print(f"Max Testing MAE:  {np.max(mae_test):.6f}")
```
