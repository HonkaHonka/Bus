import pandas as pd
import os
import random
from datetime import datetime, timedelta

def simulate_bus_data():
    """
    Simulates historical bus trip data with realistic, cumulative latency
    based on master GTFS files.
    """
    # --- 1. CONFIGURATION ---
    
    # --- Paths ---
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    master_data_path = os.path.join(script_dir, 'master_data')
    output_path = os.path.join(script_dir, 'historical_data')

    # --- Simulation Parameters ---
    TRIPS_TO_SIMULATE = 5000  # Number of trips to include in our simulation
    
    # --- Latency Rules ---
    # Rush Hour Windows (Military Time)
    MORNING_RUSH_START, MORNING_RUSH_END = 7, 9
    EVENING_RUSH_START, EVENING_RUSH_END = 16, 18
    
    # Delay multipliers (e.g., 1.5 means 50% more delay)
    RUSH_HOUR_DELAY_MULTIPLIER = 1.8
    OFF_PEAK_DELAY_MULTIPLIER = 1.2
    
    # Chance of a major random event (e.g., traffic jam)
    RANDOM_EVENT_CHANCE = 0.02 # 2% chance per stop
    RANDOM_EVENT_MIN_DELAY_SECS = 120 # 2 minutes
    RANDOM_EVENT_MAX_DELAY_SECS = 480 # 8 minutes

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created directory: {output_path}")

    # --- 2. LOAD MASTER DATA ---
    print("Loading master data files...")
    try:
        trips_df = pd.read_csv(os.path.join(master_data_path, 'master_trips.csv'), dtype=str)
        stop_times_df = pd.read_csv(os.path.join(master_data_path, 'master_stop_times.csv'), dtype=str)
    except FileNotFoundError as e:
        print(f"Error: Could not find master data files. Make sure they are in '{master_data_path}'.")
        print(f"Details: {e}")
        return

    # --- 3. PREPARE FOR SIMULATION ---
    
    # Convert time strings to a format we can use for calculations
    # This handles times like "25:30:00" which occur in transit schedules
    def parse_time(time_str):
        try:
            parts = list(map(int, time_str.split(':')))
            return timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])
        except (ValueError, IndexError):
            return None

    print("Preprocessing stop times...")
    stop_times_df['arrival_time_td'] = stop_times_df['arrival_time'].apply(parse_time)
    
    # Select a random sample of trips to simulate
    if len(trips_df) > TRIPS_TO_SIMULATE:
        trips_to_simulate = trips_df.sample(n=TRIPS_TO_SIMULATE, random_state=42)
    else:
        trips_to_simulate = trips_df

    print(f"Beginning simulation for {len(trips_to_simulate)} trips...")
    
    # This list will store the results for each stop in our simulation
    historical_records = []

    # --- 4. RUN THE SIMULATION LOOP ---
    for _, trip in trips_to_simulate.iterrows():
        trip_id = trip['trip_id']
        route_id = trip['route_id']
        
        # Get the schedule for this specific trip
        trip_schedule = stop_times_df[stop_times_df['trip_id'] == trip_id].sort_values(by='stop_sequence')
        
        # Skip trips with no valid schedule
        if trip_schedule.empty or trip_schedule['arrival_time_td'].isnull().any():
            continue

        current_cumulative_delay_seconds = 0
        
        # Iterate through each stop in the trip's schedule
        for i in range(len(trip_schedule)):
            stop_info = trip_schedule.iloc[i]
            
            # --- Calculate Latency for this segment ---
            delay_for_this_segment = 0
            
            # Determine if this stop is in rush hour
            scheduled_arrival_hour = stop_info['arrival_time_td'].total_seconds() / 3600 % 24
            is_rush_hour = (MORNING_RUSH_START <= scheduled_arrival_hour < MORNING_RUSH_END) or \
                           (EVENING_RUSH_START <= scheduled_arrival_hour < EVENING_RUSH_END)
            
            # Apply base delay based on time of day
            multiplier = RUSH_HOUR_DELAY_MULTIPLIER if is_rush_hour else OFF_PEAK_DELAY_MULTIPLIER
            delay_for_this_segment += random.uniform(0, 30) * multiplier # Base random delay

            # Apply major random event chance
            if random.random() < RANDOM_EVENT_CHANCE:
                delay_for_this_segment += random.randint(RANDOM_EVENT_MIN_DELAY_SECS, RANDOM_EVENT_MAX_DELAY_SECS)

            # A small chance for the bus to be early
            if random.random() < 0.15: # 15% chance to catch up a bit
                delay_for_this_segment -= random.uniform(0, 45)

            # Update the total cumulative delay
            current_cumulative_delay_seconds += delay_for_this_segment
            # Ensure delay doesn't become "too" negative (a bus can't be excessively early)
            current_cumulative_delay_seconds = max(-120, current_cumulative_delay_seconds)
            
            # --- Record the results ---
            actual_arrival_td = stop_info['arrival_time_td'] + timedelta(seconds=current_cumulative_delay_seconds)
            
            historical_records.append({
                'trip_id': trip_id,
                'route_id': route_id,
                'stop_id': stop_info['stop_id'],
                'stop_sequence': stop_info['stop_sequence'],
                'scheduled_arrival': stop_info['arrival_time'],
                'actual_arrival': f"{int(actual_arrival_td.total_seconds() // 3600):02d}:{int((actual_arrival_td.total_seconds() % 3600) // 60):02d}:{int(actual_arrival_td.total_seconds() % 60):02d}",
                'delay_seconds': int(current_cumulative_delay_seconds),
                'time_of_day': 'rush_hour' if is_rush_hour else 'off_peak'
            })

    # --- 5. SAVE THE FINAL DATASET ---
    print("Simulation finished. Creating final dataset...")
    final_df = pd.DataFrame(historical_records)
    output_filepath = os.path.join(output_path, 'historical_training_data.csv')
    final_df.to_csv(output_filepath, index=False)
    
    print(f"\nSuccess! Your training data has been saved to:\n{output_filepath}")
    print(f"It contains {len(final_df)} records.")

if __name__ == "__main__":
    simulate_bus_data()