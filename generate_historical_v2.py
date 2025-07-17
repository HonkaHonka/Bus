# generate_historical_v2.py
import pandas as pd
import sqlite3
import random
import os
import time

# --- Configuration ---
ROUTES_KNOWLEDGE_BASE_DB = "all_routes_bronx.db"
OUTPUT_CSV_PATH = os.path.join('historical_data', 'historical_training_data_v2.csv')
NUM_TRIPS_TO_SIMULATE = 50000 # We want a large dataset for our AI

def generate_rich_historical_data():
    """
    Generates a rich historical dataset for AI training, using the
    pre-calculated routes knowledge base.
    """
    print("--- Starting V2 Historical Data Generation ---")
    
    # --- 1. Load the Knowledge Base ---
    print(f"Loading routes from Knowledge Base: '{ROUTES_KNOWLEDGE_BASE_DB}'")
    try:
        conn_routes = sqlite3.connect(ROUTES_KNOWLEDGE_BASE_DB)
        # Load a large, random sample of routes to form our historical data
        routes_df = pd.read_sql_query(
            "SELECT * FROM routes ORDER BY RANDOM() LIMIT ?", 
            conn_routes, params=(NUM_TRIPS_TO_SIMULATE,)
        )
        conn_routes.close()
    except Exception as e:
        print(f"FATAL ERROR: Could not read from '{ROUTES_KNOWLEDGE_BASE_DB}'. {e}")
        return
        
    print(f"Simulating {len(routes_df)} trips to create the training data...")
    
    # This list will hold every single data point from our simulation
    all_records = []
    start_time = time.time()
    
    # --- 2. Run the Simulation Loop ---
    for _, route_row in routes_df.iterrows():
        path = route_row['shortest_path_stops'].split('->')
        total_distance = route_row['total_distance_km']
        total_stops = len(path)
        
        if total_stops < 2: continue

        # --- Simulate a single trip ---
        current_cumulative_delay_seconds = 0
        distance_traveled = 0.0

        for i in range(total_stops):
            current_stop_id = path[i]
            
            # --- "Logical Latency" Simulation ---
            # Determine if this stop is in rush hour (simplified)
            # We'll use a random hour for each trip to get diverse data
            simulated_hour = random.randint(0, 23)
            is_rush_hour = (7 <= simulated_hour < 10) or (16 <= simulated_hour < 19)
            
            # Add a small delay for this segment
            segment_delay = random.uniform(0, 45) * (1.8 if is_rush_hour else 1.2)
            # Small chance of a major random event
            if random.random() < 0.015:
                segment_delay += random.randint(120, 480)
            # Small chance to recover some time
            if random.random() < 0.20:
                segment_delay -= random.uniform(0, 50)
            
            current_cumulative_delay_seconds += segment_delay
            current_cumulative_delay_seconds = max(0, current_cumulative_delay_seconds) # Delay can't be negative in this simple sim
            
            # --- Record the state at this stop ---
            # We create a record *before* updating the state for the next stop
            all_records.append({
                'trip_id': f"TRIP_{route_row['start_id']}_to_{route_row['end_id']}",
                'route_id': f"RT_{route_row['start_id']}", # Synthetic route_id
                'stop_id': current_stop_id,
                'stop_sequence': i + 1,
                'current_delay': int(current_cumulative_delay_seconds),
                'time_of_day': 'rush_hour' if is_rush_hour else 'off_peak',
                'total_trip_distance': round(total_distance, 2),
                'stops_in_trip': total_stops,
                'progress_ratio': round(distance_traveled / total_distance, 3) if total_distance > 0 else 0
            })
            
            # Find distance to next stop to update distance_traveled for the *next* iteration
            if i + 1 < total_stops:
                # This is a simplification; in reality, we'd look up the segment distance.
                # For speed, we'll assume each segment is an equal fraction of the total distance.
                distance_traveled += total_distance / (total_stops - 1)


    # --- 3. Process and Save the Final Dataset ---
    print(f"\nSimulation complete. Generated {len(all_records)} records in {time.time() - start_time:.2f} seconds.")
    print("Processing final dataset...")
    
    # Convert list of dicts to a DataFrame
    final_df = pd.DataFrame(all_records)
    
    # Create the target variable: the final delay at the end of each trip
    final_df['final_delay'] = final_df.groupby('trip_id')['current_delay'].transform('last')

    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the file
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n--- Success! ---")
    print(f"Rich historical dataset saved to: '{OUTPUT_CSV_PATH}'")


if __name__ == "__main__":
    generate_rich_historical_data()