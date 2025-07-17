# check_status.py
import sqlite3
import pandas as pd
import os
import time
from datetime import datetime

DB_NAME = "live_bus_data.db"
DB_PATH = os.path.join(os.path.dirname(__file__), DB_NAME)

def check_live_status():
    """Reads the live status database and displays the current state of all buses."""
    
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file '{DB_NAME}' not found.")
        print("Hint: Is live_nodes.py running in another terminal?")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        # Use pandas to run the SQL query and get the result in a nicely formatted table
        # We select the most important columns to display in our dashboard.
        df = pd.read_sql_query(
            "SELECT trip_id, route_id, current_delay, current_lat, current_lon, last_updated FROM live_status ORDER BY trip_id", 
            conn
        )
        conn.close()

        if df.empty:
            print("No active buses found in the database. Waiting for live_nodes.py to send data...")
            return

        # Clear the terminal screen for a clean, real-time view
        # 'cls' is for Windows, 'clear' is for macOS/Linux
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("--- Current Status of Live Buses (Press Ctrl+C to Stop) ---")
        # .to_string() ensures that pandas prints the full table without shortening it
        print(df.to_string())
        print("\nLast dashboard update:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        while True:
            check_live_status()
            time.sleep(2) # Refresh the dashboard every 2 seconds
    except KeyboardInterrupt:
        print("\nStatus checker stopped.")