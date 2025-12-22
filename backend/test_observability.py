import requests
import json

base_url = "http://localhost:8000"

def call_endpoint(path):
    try:
        url = f"{base_url}{path}"
        print(f"Calling {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            print("Success!")
            try:
                data = response.json()
                print(json.dumps(data, indent=2)[:500] + ("..." if len(str(data)) > 500 else "")) # Truncate for display
            except Exception as e:
                print(f"Error parsing JSON: {e}")
        else:
            print(f"Error: Status code {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    call_endpoint("/api/observability/header")
    call_endpoint("/api/observability/learning_health?limit=5")
    call_endpoint("/api/observability/user_skills?top_k=3")
