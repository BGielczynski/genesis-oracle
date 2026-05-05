import numpy as np
import os

# Set Keras backend to JAX before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers, ops

def slice_signal(signal, window_size=50, stride=1):
    """
    Slices a 1D signal into overlapping windows.
    
    Parameters:
    - signal: 1D numpy array.
    - window_size: Number of timesteps per window (default 50).
    - stride: Step size between windows.
    
    Returns:
    - A 2D numpy array of shape (num_windows, window_size).
    """
    # Using numpy's sliding_window_view for efficient windowing (introduced in 1.20)
    # This creates a view into the original array.
    windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)
    
    # Apply stride if necessary (default is 1)
    if stride > 1:
        windows = windows[::stride]
        
    return windows

def prepare_datasets(file_path="data/corrupted_signal.npy", window_size=50, split_period=60, points_per_period=1000):
    """
    Loads the signal, splits it into training and testing sets, and applies windowing.
    
    Parameters:
    - file_path: Path to the .npy file containing the 1D signal.
    - window_size: Number of timesteps per window.
    - split_period: The period number where the split occurs (default 60).
    - points_per_period: Number of data points in one period (default 1000).
    
    Returns:
    - X_train: Windowed training data (normal data).
    - X_test: Windowed testing data (contains anomaly).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file '{file_path}' not found. Please run 'src/data_generator.py' first.")

    # Load the 1D signal
    signal = np.load(file_path)
    
    # Calculate split index
    # Normal data is before period 60
    split_idx = split_period * points_per_period
    
    train_signal = signal[:split_idx]
    test_signal = signal[split_idx:]
    
    print(f"Original signal length: {len(signal)}")
    print(f"Training signal length: {len(train_signal)} (up to period {split_period})")
    print(f"Testing signal length: {len(test_signal)} (rest)")
    
    # Create overlapping windows
    X_train = slice_signal(train_signal, window_size=window_size)
    X_test = slice_signal(test_signal, window_size=window_size)
    
    return X_train, X_test

class SignalCompression(layers.Layer):
    """
    Custom layer to compress a 1D windowed signal using Conv1D layers.
    Utilizes local patterns and reduces temporal dimension.
    """
    def __init__(self, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        # Conv-Blocks to reduce (Batch, 50, 1) -> (Batch, 25, 16) -> (Batch, 13, 32)
        self.conv1 = layers.Conv1D(filters=16, kernel_size=3, strides=2, padding="same", activation="relu")
        self.conv2 = layers.Conv1D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu")
        self.flatten = layers.Flatten()
        self.bottleneck = layers.Dense(latent_dim, activation="relu")

    def call(self, inputs):
        # Input shape: (batch_size, window_size) -> (batch_size, 50, 1)
        x = ops.expand_dims(inputs, axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.bottleneck(x)

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config

class SignalExpansion(layers.Layer):
    """
    Custom layer to expand the latent representation using Conv1DTranspose.
    Reconstructs the original 1D signal sequence.
    """
    def __init__(self, output_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        # Map back to (Batch, 13 * 32) then reshape to (Batch, 13, 32)
        self.dense = layers.Dense(13 * 32, activation="relu")
        self.reshape = layers.Reshape((13, 32))
        # Up-sampling Blöcke: (13, 32) -> (26, 16) -> (52, 1)
        self.deconv1 = layers.Conv1DTranspose(filters=16, kernel_size=3, strides=2, padding="same", activation="relu")
        self.deconv2 = layers.Conv1DTranspose(filters=1, kernel_size=3, strides=2, padding="same")
        # Crop 52 back to 50
        self.cropping = layers.Cropping1D(cropping=(0, 2))

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.cropping(x)
        # Final shape: (Batch, 50, 1) -> (Batch, 50)
        return ops.squeeze(x, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim})
        return config

class PhysicsAutoencoder(keras.Model):
    """
    Upgraded Convolutional Autoencoder for physical signal flows.
    """
    def __init__(self, window_size=50, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.encoder = SignalCompression(latent_dim=latent_dim)
        self.decoder = SignalExpansion(output_dim=window_size)

    def call(self, inputs):
        latent = self.encoder(inputs)
        reconstructed = self.decoder(latent)
        return reconstructed

if __name__ == "__main__":
    # Example usage and verification
    try:
        # Check if the data directory exists, if not, try to run data_generator (optional but helpful)
        data_file = "data/corrupted_signal.npy"
        
        X_train, X_test = prepare_datasets(data_file)
        
        print("\nData preparation successful!")
        print(f"X_train shape: {X_train.shape}  (Training windows)")
        print(f"X_test shape:  {X_test.shape}   (Testing windows)")
        
        # Verify overlap:
        # If stride=1, consecutive windows should differ by only one shift.
        if X_train.shape[0] > 1:
            is_overlapping = np.array_equal(X_train[0, 1:], X_train[1, :-1])
            print(f"Windows are overlapping: {is_overlapping}")

        # Demonstrate the custom layer (without training)
        compression_layer = SignalCompression(latent_dim=8)
        
        # Take a small batch of training data
        sample_batch = X_train[:5].astype("float32")
        latent_representation = compression_layer(sample_batch)
        
        print("\nCustom Layer Verification:")
        print(f"Input batch shape:  {sample_batch.shape}")
        print(f"Latent batch shape: {latent_representation.shape}")
        print(f"First latent vector example:\n{latent_representation[0]}")

        # Demonstrate the full PhysicsAutoencoder (without training)
        autoencoder = PhysicsAutoencoder(window_size=50, latent_dim=8)
        reconstruction = autoencoder(sample_batch)
        
        print("\nPhysicsAutoencoder Verification:")
        print(f"Reconstruction shape: {reconstruction.shape}")
        print(f"Original first sample (head):\n{sample_batch[0][:5]}...")
        print(f"Reconstructed first sample (head):\n{reconstruction[0][:5]}...")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
