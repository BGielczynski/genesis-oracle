import numpy as np
import matplotlib.pyplot as plt
import os

def generate_signal():
    t = np.linspace(0, 10, 1000)
    # Base sine wave
    signal = np.sin(2 * np.pi * 0.5 * t) + 0.2 * np.random.normal(size=len(t))
    
    # Inject high-frequency clipping artifact at a random location
    # Requirement: "at a random timestep, inject an ugly, high-frequency clipping artifact (amplitude saturation)"
    idx = np.random.randint(200, 800)
    width = 50
    signal[idx : idx + width] = np.clip(signal[idx : idx + width] + 2.0 * np.sin(50 * t[idx : idx + width]), -0.8, 0.8)
    
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, color='blue')
    plt.title("System Telemetry Stream")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Save output plot as data/audit_target.png
    if not os.path.exists("data"):
        os.makedirs("data")
    
    plt.savefig("data/audit_target.png")
    # Requirement: "without printing the timestamp to the terminal"
    # (We just don't print idx or t[idx])

if __name__ == "__main__":
    generate_signal()
