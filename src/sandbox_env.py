import numpy as np

class ThermalDampener:
    def __init__(self, initial_kappa=10.0):
        self.kappa = initial_kappa
        self.target_temp = 300.0  # Kelvin
        self.current_temp = 350.0 # Starting volatile state
        self.history = []

    def step(self):
        # Simple physical model: Temperature change depends on Kappa
        # Optimal Kappa around 5.0 leads to stability
        # High Kappa (> 8.0) -> Boiling/Exploding
        # Low Kappa (< 2.0) -> Freezing
        
        noise = np.random.normal(0, 0.5)
        # The higher the kappa, the more heat is added
        self.current_temp += (self.kappa - 5.0) * 2.0 + noise
        self.history.append(self.current_temp)
        
        if self.current_temp > 400:
            return "BOILING"
        elif self.current_temp < 250:
            return "FREEZING"
        elif 295 <= self.current_temp <= 305:
            return "PERFECT"
        else:
            return "UNSTABLE"

    def get_status(self):
        return {
            "temperature": round(self.current_temp, 2),
            "kappa": round(self.kappa, 2)
        }
