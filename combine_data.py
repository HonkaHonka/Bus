import pandas as pd
import os

def combine_gtfs_files():
    """
    Reads GTFS text files from multiple subdirectories, combines them,
    and saves the output as master CSV files.
    """
    # Define paths relative to the script's location
    script_dir = os.path.dirname(__file__)
    base_data_path = os.path.join(script_dir, 'data')
    output_path = os.path.join(script_dir, 'master_data')

    # Ensure the output directory exists
    if not os.path.exists(output_path):
        print(f"Creating output directory: {output_path}")
        os.makedirs(output_path)

    # The names of the folders inside the 'data' directory
    data_folders = [
        'gtfs_b',
        'gtfs_bx',
        'gtfs_m',
        'gtfs_q',
        'gtfs_si',
        'gtfs_busco'
    ]

    # The GTFS files we want to combine
    files_to_combine = [
        'stops.txt',
        'routes.txt',
        'trips.txt',
        'stop_times.txt',
        'shapes.txt',
        'calendar.txt',
        'calendar_dates.txt',
        'agency.txt'
    ]

    print("Starting data combination process...")

    # Process each file type one by one
    for filename in files_to_combine:
        list_of_dfs = []
        print(f"\nProcessing {filename}...")

        # Go through each borough/company folder
        for folder in data_folders:
            file_path = os.path.join(base_data_path, folder, filename)
            
            try:
                # Check if the file actually exists before trying to read
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, dtype=str) # Read all data as strings to avoid type conflicts
                    list_of_dfs.append(df)
                    print(f"  - Successfully read from: {file_path}")
                else:
                    print(f"  - Warning: File not found and will be skipped: {file_path}")

            except Exception as e:
                print(f"  - Error reading {file_path}: {e}")

        # If we have successfully read at least one file, combine and save
        if list_of_dfs:
            # Combine all the dataframes for the current file type
            combined_df = pd.concat(list_of_dfs, ignore_index=True)
            
            # Create the new filename for the master file
            output_filename = f"master_{filename.replace('.txt', '.csv')}"
            output_filepath = os.path.join(output_path, output_filename)
            
            # Save the combined dataframe to a CSV file
            combined_df.to_csv(output_filepath, index=False)
            print(f"-> Successfully created master file: {output_filepath}")
        else:
            print(f"-> No data found for {filename}. No master file created.")

    print("\nData combination process finished!")

if __name__ == "__main__":
    combine_gtfs_files()