
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

url = "https://integrate.api.nvidia.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

payload = {
    "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "messages": [
        {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 100,
    "temperature": 0.2,
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=60,
)

print("HTTP Status:", response.status_code)
print("Response:", response.text)
