# setup_postgres_tables.py (V3 - Definitive)
from sqlalchemy import create_engine, text
from config import DB_CONFIG

def setup_database_tables():
    print("--- Setting up database tables (V3) in PostgreSQL ---")
    try:
        db_url = (f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
                  f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
        engine = create_engine(db_url)
        
        with engine.connect() as connection:
            print("Resetting 'live_status' table...")
            connection.execute(text('DROP TABLE IF EXISTS live_status;'))
            create_live_table_query = """
            CREATE TABLE live_status (
                trip_id TEXT PRIMARY KEY,
                route_id TEXT NOT NULL,
                start_id TEXT NOT NULL, -- NEW
                end_id TEXT NOT NULL,   -- NEW
                stop_sequence INTEGER NOT NULL,
                current_delay INTEGER NOT NULL,
                time_of_day TEXT NOT NULL,
                current_lat REAL NOT NULL, 
                current_lon REAL NOT NULL,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """
            connection.execute(text(create_live_table_query))
            print("-> 'live_status' table is ready.")

            create_history_table_query = """
            CREATE TABLE IF NOT EXISTS trip_history (
                record_id SERIAL PRIMARY KEY, trip_id TEXT NOT NULL, route_id TEXT NOT NULL,
                start_station_id TEXT NOT NULL, end_station_id TEXT NOT NULL,
                final_delay INTEGER NOT NULL, completed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """
            connection.execute(text(create_history_table_query))
            print("-> 'trip_history' table is ready.")
            connection.commit()
        print("\nDatabase setup complete.")
    except Exception as e:
        print(f"\nDATABASE ERROR: {e}")

if __name__ == "__main__":
    setup_database_tables()