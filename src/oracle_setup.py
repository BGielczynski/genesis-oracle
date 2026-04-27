import os

# Set Keras backend to JAX before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import keras
import jax
import numpy as np

def verify_setup():
    print(f"Keras version: {keras.__version__}")
    print(f"Keras backend: {keras.backend.backend()}")
    print(f"JAX version: {jax.__version__}")
    
    # Simple JAX test
    x = jax.numpy.array([1.0, 2.0, 3.0])
    print(f"JAX array test: {x}")

    # Keras random tensor test
    random_tensor = keras.random.normal(shape=(2, 2))
    print(f"Keras random tensor:\n{random_tensor}")
    print(f"Keras tensor type: {type(random_tensor)}")

if __name__ == "__main__":
    verify_setup()
