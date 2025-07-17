# create_mission_list.py
from sqlalchemy import create_engine, text
from config import DB_CONFIG

def create_mission_table():
    """
    Selects a smaller, optimized subset of routes for the live simulation
    to choose from, making startup much faster.
    """
    MISSION_TABLE = "mission_routes"
    SOURCE_TABLE = "all_city_routes"
    NUM_MISSIONS = 5000 # Select 5000 good routes for our mission pool
    
    print(f"--- Creating optimized '{MISSION_TABLE}' table ---")
    
    try:
        db_url = (f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
                  f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
        engine = create_engine(db_url)
        
        with engine.connect() as connection:
            # This SQL command creates the new table by selecting a random sample
            # of good, medium-length routes from the main knowledge base.
            sql = text(f"""
                DROP TABLE IF EXISTS {MISSION_TABLE};
                CREATE TABLE {MISSION_TABLE} AS
                SELECT * FROM {SOURCE_TABLE}
                WHERE LENGTH(shortest_path_stops) > 100 
                ORDER BY RANDOM() 
                LIMIT {NUM_MISSIONS};
            """)
            connection.execute(sql)
            connection.commit()
            
        print("Mission list table created successfully.")
        
    except Exception as e:
        print(f"\nDATABASE ERROR: {e}")

if __name__ == "__main__":
    create_mission_table()