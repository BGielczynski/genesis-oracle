import os
import json
from pydantic import BaseModel, Field
from google import genai
from sandbox_env import ThermalDampener

# 1. Define the strict structured contract using Pydantic
class ControlDecision(BaseModel):
    system_state: str = Field(description="Must be 'FREEZING', 'BOILING', or 'PERFECT'")
    adjustment_action: str = Field(description="Must be 'INCREASE', 'DECREASE', or 'HOLD'")
    delta_value: float = Field(description="The exact numerical change to apply to Kappa")
    confidence_score: float

def run_game_loop():
    api_key = os.environ.get("GEMINI_API_KEY") or "AIzaSyDBWnSS029pt_exufvk3HBndWc3zbMM4PE"
    client = genai.Client(api_key=api_key)
    env = ThermalDampener(initial_kappa=10.0) # Volatile start
    
    print(f"--- Starting Game Loop ---")
    print(f"Initial State: {env.get_status()}\n")

    for turn in range(1, 6):
        status = env.get_status()
        state_label = env.step() # Advance physics
        
        prompt = f"""SYSTEM STATUS:
Current Temperature: {status['temperature']}K
Current Kappa: {status['kappa']}
Target Temperature: 300K

PHYSICS RULE: 
- If Temp > 300K, you MUST DECREASE Kappa.
- If Temp < 300K, you MUST INCREASE Kappa.
- Kappa around 5.0 is neutral. Currently Kappa is {status['kappa']}, which causes the temperature to change by {(status['kappa'] - 5.0) * 2.0}K per step.

MISSION:
Calculate the necessary 'delta_value' (always positive) and 'adjustment_action' to reach the target.
Provide your response as a JSON object matching the ControlDecision schema."""

        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ControlDecision,
                }
            )
            
            decision = response.parsed
            print(f"Turn {turn}:")
            print(f"  Model Diagnosis: {decision.system_state}")
            print(f"  Action: {decision.adjustment_action} by {decision.delta_value}")
            
            # Apply adjustment
            if decision.adjustment_action == "INCREASE":
                env.kappa += decision.delta_value
            elif decision.adjustment_action == "DECREASE":
                env.kappa -= decision.delta_value
            
            new_status = env.get_status()
            print(f"  New State: {new_status}")
            
            if new_status['temperature'] >= 295 and new_status['temperature'] <= 305:
                print(f"  >>> SUCCESS: SYSTEM STABILIZED IN {turn} TURNS <<<\n")
                break
            print("-" * 30)

        except Exception as e:
            print(f"Error in turn {turn}: {e}")
            break

if __name__ == "__main__":
    run_game_loop()
