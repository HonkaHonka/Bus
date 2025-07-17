# setup_stops_table.py
import pandas as pd
from sqlalchemy import create_engine
from config import DB_CONFIG

def populate_stops_directory():
    """
    Reads the master stops file and populates a new, clean table
    in PostgreSQL for fast lookups.
    """
    TABLE_NAME = "stops_directory"
    STOPS_FILE = "master_data/master_stops.csv"
    
    print(f"--- Populating '{TABLE_NAME}' table in PostgreSQL ---")
    
    try:
        # Load and clean the master stops file
        stops_df = pd.read_csv(STOPS_FILE, dtype=str)
        stops_df.drop_duplicates(subset='stop_id', inplace=True)
        stops_df = stops_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
        print(f"Loaded and cleaned {len(stops_df)} unique stops.")

        # Connect to the database
        db_url = (f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
                  f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
        engine = create_engine(db_url)

        # Upload the data, replacing the table if it already exists
        stops_df.to_sql(TABLE_NAME, engine, if_exists='replace', index=False)
        
        print(f"Successfully created and populated the '{TABLE_NAME}' table.")
        
    except Exception as e:
        print(f"\nDATABASE ERROR: Could not populate stops table. {e}")

if __name__ == "__main__":
    populate_stops_directory()