import requests
import json
import time

# --- IMPORTANT ---
# 1. Replace this with the HTTPS URL provided by your running ngrok terminal (Terminal 2).
#    Domain copied from your Ngrok dashboard: molly-breedable-kent.ngrok-free.dev
BOT_ENDPOINT = "https://molly-breedable-kent.ngrok-free.dev/quiz" 

# 2. Use your actual quiz credentials
YOUR_EMAIL = "24f1000999@ds.study.iitm.ac.in"
YOUR_SECRET = "elephant" # Must match your .env file
INITIAL_QUIZ_URL = "https://tds-llm-analysis.s-anand.net/demo" 

# --- The Payload ---
payload = {
    "email": YOUR_EMAIL,
    "secret": YOUR_SECRET,
    "url": INITIAL_QUIZ_URL
}

print(f"--- Triggering Quiz Agent ---")
print(f"Target URL: {BOT_ENDPOINT}")
print(f"Payload: {payload['url']}")
print(f"-----------------------------")

try:
    response = requests.post(BOT_ENDPOINT, json=payload)
    response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
    
    print("\n[API Response]")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=4))
    print("\n--- Monitoring Terminal 1 Logs ---")
    print("The agent is now running in the background. Watch the other terminal!")
    
    # Wait for the process to finish
    time.sleep(5) 
    
except requests.exceptions.RequestException as e:
    print(f"\n[ERROR] Failed to contact the bot via ngrok.")
    print(f"Details: {e}")
    print("--> CHECK: Is your 'python main.py' running? Is your 'ngrok http 8000' running? Is the BOT_ENDPOINT correct?")