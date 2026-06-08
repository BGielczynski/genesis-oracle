import os
from google import genai

def ping_oracle():
    # API Key provided by user
    api_key = os.environ.get("GEMINI_API_KEY") or "AIzaSyDBWnSS029pt_exufvk3HBndWc3zbMM4PE"
    client = genai.Client(api_key=api_key)
    try:
        print("Available models:")
        for m in client.models.list():
            print(f" - {m.name}")
        
        # Using gemini-3.5-flash as per instructions
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents="Explain the difference between a stateful NumPy random generation process and a stateless JAX PRNG split operation in exactly one highly sarcastic sentence."
        )
        print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    ping_oracle()
