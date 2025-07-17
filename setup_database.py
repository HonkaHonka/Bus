# setup_database.py (Upgraded Version)
import sqlite3
import os

DB_NAME = "live_bus_data.db"
DB_PATH = os.path.join(os.path.dirname(__file__), DB_NAME)

def create_database():
    """Creates/recreates the database with location-aware columns."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed old database '{DB_NAME}'.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # We add current_lat and current_lon to store the bus's precise position.
    create_table_query = """
    CREATE TABLE live_status (
        trip_id TEXT PRIMARY KEY,
        route_id TEXT NOT NULL,
        stop_sequence INTEGER NOT NULL,
        current_delay INTEGER NOT NULL,
        time_of_day TEXT NOT NULL,
        current_lat REAL NOT NULL, 
        current_lon REAL NOT NULL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    cursor.execute(create_table_query)
    conn.commit()
    print(f"Database '{DB_NAME}' created with location-aware 'live_status' table.")
    conn.close()

if __name__ == "__main__":
    create_database()