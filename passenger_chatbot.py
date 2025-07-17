# passenger_chatbot.py
import requests
import json
import math

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"

def get_passenger_update(trip_id: str):
    """Gets a prediction and formats it into a simple status for a passenger."""
    
    print(f"\nChecking status for trip: {trip_id}...")
    
    try:
        # We call the original endpoint, as passengers don't need regulation advice.
        response = requests.post(f"{API_BASE_URL}/predict_live_v2", json={"trip_id": trip_id})
        
        if response.status_code == 200:
            data = response.json()
            
            # --- Format the data into friendly text ---
            delay_seconds = data['predicted_final_delay_seconds']
            delay_minutes = round(delay_seconds / 60)
            
            if delay_minutes > 2:
                status_text = f"is running about {delay_minutes} minutes late."
            elif delay_minutes < -2:
                status_text = f"is running about {abs(delay_minutes)} minutes early."
            else:
                status_text = "is running on time."
            
            stops_remaining = len(data['static_route_details']['path_remaining'])
            
            print("\n--- Trip Status ---")
            print(f"Your bus {status_text}")
            if stops_remaining > 0:
                print(f"There are {stops_remaining} stops until your final destination.")
            else:
                print("This trip has finished.")
            print("-------------------")

        else:
            print(f"Error checking status: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to the API. Is 'uvicorn api:app' running?")

if __name__ == "__main__":
    print("--- Welcome to the TransitPulse Passenger Chatbot ---")
    print("You can get status updates for any active trip.")
    print("Hint: Run check_status.py in another terminal to see active trip IDs.")
    
    while True:
        trip_input = input("\nPlease enter the Trip ID you want to check (or 'quit' to exit): ")
        if trip_input.lower() == 'quit':
            break
        if not trip_input:
            continue
            
        get_passenger_update(trip_input.strip())