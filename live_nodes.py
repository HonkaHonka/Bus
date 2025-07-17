# live_nodes.py (V8.2 - Definitive Final Version)
import pandas as pd
import random
import time
import os
from datetime import datetime
from sqlalchemy import create_engine, text
from config import DB_CONFIG

# --- Configuration ---
NUM_BUSES_TO_SIMULATE = 15
UPDATE_INTERVAL_SECONDS = 2
SEGMENT_TRAVEL_TIME_SECONDS = 120

def get_new_mission(conn):
    query = text("SELECT * FROM mission_routes ORDER BY RANDOM() LIMIT 1")
    result = conn.execute(query).mappings().first()
    return pd.Series(result) if result else None

def run_city_wide_nodes():
    print("--- Starting CITY-WIDE Live Bus Node Simulation (V8.2 - Final) ---")
    
    # --- Load Data and Connect to DB (Unchanged) ---
    try:
        stops_df = pd.read_csv(os.path.join('master_data', 'master_stops.csv'), dtype=str)
        stops_df.drop_duplicates(subset='stop_id', inplace=True); stops_df.set_index('stop_id', inplace=True)
        stops_df['stop_lat'] = pd.to_numeric(stops_df['stop_lat'], errors='coerce')
        stops_df['stop_lon'] = pd.to_numeric(stops_df['stop_lon'], errors='coerce')
        stops_coords = stops_df[['stop_lat', 'stop_lon']].dropna().to_dict('index')
        valid_stop_ids = set(stops_coords.keys())
        db_url = (f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}" f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
        engine = create_engine(db_url); conn_live = engine.connect()
        print("Successfully connected to PostgreSQL.")
    except Exception as e: print(f"\nDATABASE ERROR: Could not connect or load data. {e}"); return
        
    # --- Initialize Buses (Unchanged) ---
    active_buses = {}
    print("Initializing city-wide bus fleet (FAST)...")
    with conn_live.begin():
        while len(active_buses) < NUM_BUSES_TO_SIMULATE:
            mission = get_new_mission(conn_live)
            if mission is None: continue
            start_id, end_id = mission['start_id'], mission['end_id']
            current_path = mission['shortest_path_stops'].split('->')
            if set(current_path).issubset(valid_stop_ids):
                trip_id = f"TRIP_{start_id}_to_{end_id}_{int(time.time())}"
                active_buses[trip_id] = { "path": current_path, "start_id": start_id, "end_id": end_id, "route_id": f"RT_{start_id}", "current_stop_index": 0, "segment_start_time": time.time(), "cumulative_delay": random.randint(-30, 90) }
                print(f"  -> Initialized: {trip_id}")
    
    # --- Main Loop ---
    print(f"\nBroadcasting updates every {UPDATE_INTERVAL_SECONDS}s. Press Ctrl+C to stop.")
    try:
        while True:
            # --- THIS IS THE FIX ---
            # Define current_real_time at the beginning of each cycle
            current_real_time = time.time()
            # --- END OF FIX ---
            
            with conn_live.begin():
                for trip_id in list(active_buses.keys()):
                    try:
                        bus = active_buses[trip_id]
                        if bus['current_stop_index'] + 1 >= len(bus['path']):
                            print(f"\n--- Trip {trip_id} finished. Assigning new mission. ---")
                            log_sql = text("""INSERT INTO trip_history (trip_id, route_id, start_station_id, end_station_id, final_delay) VALUES (:trip_id, :route_id, :start_id, :end_id, :final_delay)""")
                            conn_live.execute(log_sql, {"trip_id": trip_id, "route_id": bus['route_id'], "start_id": bus['start_id'], "end_id": bus['end_id'], "final_delay": int(bus['cumulative_delay'])})
                            conn_live.execute(text("DELETE FROM live_status WHERE trip_id = :trip_id"), {"trip_id": trip_id})
                            mission = get_new_mission(conn_live)
                            if mission is not None:
                                start_id, end_id = mission['start_id'], mission['end_id']
                                new_trip_id = f"TRIP_{start_id}_to_{end_id}_{int(time.time())}"
                                new_path = mission['shortest_path_stops'].split('->')
                                if set(new_path).issubset(valid_stop_ids):
                                    active_buses[new_trip_id] = { "path": new_path, "start_id": start_id, "end_id": end_id, "route_id": f"RT_{start_id}", "current_stop_index": 0, "segment_start_time": time.time(), "cumulative_delay": random.randint(-30, 90) }
                            del active_buses[trip_id]
                            continue
                        
                        bus['cumulative_delay'] += random.uniform(-2, 3); bus['cumulative_delay'] = max(-120, bus['cumulative_delay'])
                        current_stop_idx = bus['current_stop_index']
                        from_stop_id = bus['path'][current_stop_idx]; to_stop_id = bus['path'][current_stop_idx+1]
                        elapsed_time_in_segment = current_real_time - bus['segment_start_time']
                        if elapsed_time_in_segment >= SEGMENT_TRAVEL_TIME_SECONDS:
                            bus['current_stop_index'] += 1; bus['segment_start_time'] = time.time()
                        progress_ratio = (current_real_time - bus['segment_start_time']) / SEGMENT_TRAVEL_TIME_SECONDS
                        lat = stops_coords[from_stop_id]['stop_lat'] + (stops_coords[to_stop_id]['stop_lat'] - stops_coords[from_stop_id]['stop_lat']) * progress_ratio
                        lon = stops_coords[from_stop_id]['stop_lon'] + (stops_coords[to_stop_id]['stop_lon'] - stops_coords[from_stop_id]['stop_lon']) * progress_ratio
                        upsert_sql = text("""INSERT INTO live_status (trip_id, route_id, start_id, end_id, stop_sequence, current_delay, time_of_day, current_lat, current_lon, last_updated) VALUES (:trip_id, :route_id, :start_id, :end_id, :stop_sequence, :current_delay, :time_of_day, :current_lat, :current_lon, NOW()) ON CONFLICT (trip_id) DO UPDATE SET route_id = EXCLUDED.route_id, start_id = EXCLUDED.start_id, end_id = EXCLUDED.end_id, stop_sequence = EXCLUDED.stop_sequence, current_delay = EXCLUDED.current_delay, time_of_day = EXCLUDED.time_of_day, current_lat = EXCLUDED.current_lat, current_lon = EXCLUDED.current_lon, last_updated = NOW();""")
                        params = { "trip_id": trip_id, "route_id": bus['route_id'], "start_id": bus['start_id'], "end_id": bus['end_id'], "stop_sequence": current_stop_idx + 1, "current_delay": int(bus['cumulative_delay']), "time_of_day": "rush_hour" if (7 <= datetime.now().hour < 10) else "off_peak", "current_lat": round(lat, 6), "current_lon": round(lon, 6) }
                        conn_live.execute(upsert_sql, params)
                    except Exception as e:
                        print(f"\n--- ERROR processing trip {trip_id}: {e} ---")
                        if trip_id in active_buses: del active_buses[trip_id]
            
            print(f"--- Ecosystem updated: {datetime.now().strftime('%H:%M:%S')} (Active Buses: {len(active_buses)}) ---", end="\r")
            time.sleep(UPDATE_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n--- Simulation stopped by user. ---")
    finally:
        conn_live.close()

if __name__ == "__main__":
    run_city_wide_nodes()