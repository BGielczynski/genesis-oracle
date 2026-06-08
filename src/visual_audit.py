import os
from google import genai
from PIL import Image

def visual_audit():
    api_key = os.environ.get("GEMINI_API_KEY") or "AIzaSyDBWnSS029pt_exufvk3HBndWc3zbMM4PE"
    client = genai.Client(api_key=api_key)
    
    img_path = "data/audit_target.png"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    img = Image.open(img_path)
    
    prompt = """Act as a Visual Detective. 
Find the visual anomaly in this signal plot. 
Guess the exact pixel/X-axis region where the malfunction happened.
Write a short, funny poem mocking the engineering team that allowed this bug to pass."""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[prompt, img]
        )
        print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    visual_audit()
