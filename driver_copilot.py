# driver_copilot.py
import requests
import time
import os

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
POLL_INTERVAL_SECONDS = 10 # Check for new advice every 10 seconds

# A simple (and cross-platform) way to simulate speech.
# For a real app, you would use a proper text-to-speech library like gTTS or pyttsx3.
def speak(text: str):
    """A simple function to simulate the AI speaking to the driver."""
    print("\n" + "="*50)
    print(f"|| [CO-PILOT VOICE]: {text}")
    print("="*50 + "\n")

def run_driver_copilot(trip_id: str):
    """Runs a continuous loop to poll for and deliver regulation advice."""
    
    print(f"--- TransitPulse Driver Co-Pilot Activated ---")
    print(f"Now monitoring trip: {trip_id}")
    print(f"Checking for new advice every {POLL_INTERVAL_SECONDS} seconds. Press Ctrl+C to stop.")
    
    last_advice = None
    
    while True:
        try:
            # We call the new, more advanced endpoint
            response = requests.post(f"{API_BASE_URL}/get_regulation_advice", json={"trip_id": trip_id})
            
            if response.status_code == 200:
                data = response.json()
                current_advice = data['regulation_advice']
                advice_level = data['advice_level']
                
                # --- The Core Logic ---
                # Only speak if the advice is new and important
                if current_advice != last_advice and advice_level == 'ACTION_REQUIRED':
                    speak(current_advice)
                    last_advice = current_advice # Remember the last thing we said
                
                # Also print a silent, continuous status update
                delay_minutes = data['predicted_final_delay_seconds'] / 60
                print(f"Status: {delay_minutes:.1f} min predicted delay. Advice Level: {advice_level}. Last Spoken: '{last_advice}'", end="\r")

            else:
                print(f"API Error: {response.status_code}", end="\r")

            time.sleep(POLL_INTERVAL_SECONDS)

        except requests.exceptions.ConnectionError:
            print("\nERROR: API connection lost. Retrying in 10 seconds...")
            time.sleep(POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\n--- Co-Pilot Deactivated ---")
            break

if __name__ == "__main__":
    # In a real app, the driver would log in to get their trip_id.
    # For our simulation, we'll just ask for it once.
    print("--- Co-Pilot Initialization ---")
    print("Hint: Run check_status.py in another terminal to find a trip to monitor.")
    trip_to_monitor = input("Please enter the Trip ID to monitor for this session: ")
    
    if trip_to_monitor:
        run_driver_copilot(trip_to_monitor.strip())