# main_app.py (The Unified TransitPulse Application)
import requests
import time
import os
import pandas as pd
import sqlite3

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
LIVE_DB_PATH = "live_bus_data.db"
POLL_INTERVAL_SECONDS = 10

# Helper function to clear the screen
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# =============================================================================
# --- "PASSENGER" PARTITION LOGIC ---
# =============================================================================
def view_schedule():
    """Queries and displays a simplified schedule of active trips."""
    print("\n--- Currently Active Trips ---")
    try:
        conn = sqlite3.connect(LIVE_DB_PATH)
        df = pd.read_sql_query("SELECT trip_id, route_id FROM live_status ORDER BY route_id", conn)
        conn.close()
        if df.empty:
            print("No buses are currently active. Please start the simulation.")
        else:
            print(df.to_string())
    except Exception as e:
        print(f"Could not fetch schedule. Is live_nodes.py running? Error: {e}")
    input("\nPress Enter to return to the menu...")

def passenger_chatbot():
    """Handles the chatbot interaction for a passenger."""
    trip_id = input("Please enter the Trip ID you want to check: ").strip()
    if not trip_id: return

    print(f"\nChecking status for trip: {trip_id}...")
    try:
        response = requests.post(f"{API_BASE_URL}/predict_live_v2", json={"trip_id": trip_id})
        if response.status_code == 200:
            data = response.json()
            delay_seconds = data['predicted_final_delay_seconds']
            delay_minutes = round(delay_seconds / 60)
            status = "on time"
            if delay_minutes > 2: status = f"about {delay_minutes} minutes late"
            elif delay_minutes < -2: status = f"about {abs(delay_minutes)} minutes early"
            
            stops_remaining = len(data['static_route_details']['path_remaining'])
            print(f"\n[CHATBOT]: Your bus is running {status}. There are {stops_remaining} stops remaining.")
        else:
            print(f"[CHATBOT]: Sorry, I couldn't get the status. Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("[CHATBOT]: I can't connect to the main server right now. Please try again later.")
    input("\nPress Enter to return to the menu...")

def run_passenger_space():
    """Main loop for the passenger partition."""
    while True:
        clear_screen()
        print("--- Welcome to the TransitPulse Passenger Portal ---")
        print("\n1. View Schedule of Active Trips")
        print("2. Ask Chatbot for Trip Status")
        print("3. Logout")
        choice = input("\nPlease choose an option: ")

        if choice == '1':
            view_schedule()
        elif choice == '2':
            passenger_chatbot()
        elif choice == '3':
            break
        else:
            print("Invalid option, please try again.")
            time.sleep(1)

# =============================================================================
# --- "DRIVER" PARTITION LOGIC ---
# =============================================================================
def speak(text: str):
    """Simulates the AI speaking to the driver."""
    print("\n" + "="*60)
    print(f"|| [CO-PILOT VOICE]: {text}")
    print("="*60)

def draw_map(path_completed: list, path_remaining: list):
    """Draws a simple text-based map of the route."""
    print("\n--- Route Map ---")
    # Join the path with a "faded" completed segment
    completed_str = " -> ".join(f"({s})" for s in path_completed)
    remaining_str = " -> ".join(path_remaining)
    
    if not path_remaining: # If the trip is over
        print(f"{completed_str} -> [TRIP COMPLETE]")
    else:
        # Place the bus icon at the junction
        print(f"{completed_str} -[BUS]-> {remaining_str}")

def run_driver_space():
    """Main loop and logic for the driver co-pilot."""
    clear_screen()
    print("--- Driver Co-Pilot Activation ---")
    trip_id = input("Please enter the Trip ID for your current route: ").strip()
    if not trip_id: return

    last_advice = None
    try:
        while True:
            clear_screen()
            print(f"--- Co-Pilot Monitoring Trip: {trip_id} ---")
            print("Status: Polling for new data...")
            
            try:
                response = requests.post(f"{API_BASE_URL}/get_regulation_advice", json={"trip_id": trip_id})
                if response.status_code == 200:
                    data = response.json()
                    current_advice = data['regulation_advice']
                    advice_level = data['advice_level']

                    # Draw the live map
                    draw_map(data['static_route_details']['path_completed'], data['static_route_details']['path_remaining'])
                    
                    # Speak new, important advice
                    if current_advice != last_advice and advice_level == 'ACTION_REQUIRED':
                        speak(current_advice)
                        last_advice = current_advice

                    # Display current status
                    delay_minutes = data['predicted_final_delay_seconds'] / 60
                    print(f"\nStatus: {delay_minutes:.1f} min predicted delay | Advice Level: {advice_level}")
                    print(f"Last vocal alert: '{last_advice}'")

                else:
                    print(f"\nAPI Error: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                print("\nERROR: API connection lost. Retrying...")
            
            print(f"\n(Updating in {POLL_INTERVAL_SECONDS} seconds... Press Ctrl+C to end shift)")
            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nShift ended. Logging out...")
        time.sleep(2)


# =============================================================================
# --- MAIN APPLICATION LOGIN ---
# =============================================================================
if __name__ == "__main__":
    while True:
        clear_screen()
        print("==================================================")
        print("      Welcome to the TransitPulse System      ")
        print("==================================================")
        print("\nPlease select your role:")
        print("\n1. Passenger")
        print("2. Driver")
        print("\n3. Exit Application")
        
        role = input("\nEnter your choice: ")
        
        if role == '1':
            run_passenger_space()
        elif role == '2':
            run_driver_space()
        elif role == '3':
            print("Exiting application. Goodbye!")
            break
        else:
            print("Invalid role. Please try again.")
            time.sleep(2)