import os
from groq import Groq

# Load API key from .env file manually
def load_env():
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

load_env()

# Create Groq client
client = Groq(api_key=os.environ["GROQ_API_KEY"])

print("Sending test message to Llama 3...")
print("-" * 40)

# Send a simple message
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",   # new — works!",   # free Llama 3 model
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer clearly and concisely."
        },
        {
            "role": "user",
            "content": "What is generative AI in 2 sentences?"
        }
    ]
)

# Print the response
answer = response.choices[0].message.content
print(answer)
print("-" * 40)
print("\n✓ Groq connection working!")