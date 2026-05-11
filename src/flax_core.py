import jax
import jax.numpy as jnp
from flax import linen as nn

class SimpleMLP(nn.Module):
    """
    Ein einfaches Multi-Layer Perceptron in Flax.
    Beachte: Dieses Objekt speichert KEINE Gewichte. Es ist lediglich ein 
    mathematischer Bauplan (Blueprint).
    """
    hidden_dims: int = 64
    output_dims: int = 1

    @nn.compact
    def __call__(self, x):
        # nn.compact erlaubt es, Layer direkt in der Vorwärtsrechnung zu definieren.
        # Die Gewichte werden erst bei der Initialisierung generiert.
        x = nn.Dense(features=self.hidden_dims)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dims)(x)
        return x

def run_demonstration():
    # --- SCHRITT 1: PRNGKey erzeugen ---
    # JAX ist deterministisch. Wir brauchen einen expliziten Zufallsschlüssel.
    key = jax.random.PRNGKey(0)
    
    # --- SCHRITT 2: Modul-Instanziierung ---
    # Das 'model' Objekt enthält nur die Architektur-Definition.
    model = SimpleMLP()
    
    # --- SCHRITT 3: Explizite Initialisierung (model.init) ---
    # Im Gegensatz zu Keras speichert Flax die Gewichte NICHT im Objekt.
    # Wir rufen .init mit einem Beispiel-Input auf. Das Ergebnis ist ein
    # 'FrozenDict', das alle Parameter (Weights/Biases) enthält.
    dummy_input = jnp.ones((1, 10))
    params = model.init(key, dummy_input)
    
    print("--- Flax State Management ---")
    print(f"Modul-Typ: {type(model)}")
    print("Die Parameter sind ein separates Objekt (stateless):")
    for layer, values in params['params'].items():
        print(f" -> {layer}: {values.keys()}")

    # --- SCHRITT 4: Expliziter Forward Pass (model.apply) ---
    # Da das Model selbst keine Daten hält, müssen wir die 'params' beim
    # Aufruf von .apply explizit übergeben.
    # Keras-Stil: model(x) | Flax-Stil: model.apply(params, x)
    output = model.apply(params, dummy_input)
    
    print("\n--- Forward Pass Ergebnis ---")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print(f"Stateless Execution erfolgreich: Output berechnet.")

if __name__ == "__main__":
    run_demonstration()
