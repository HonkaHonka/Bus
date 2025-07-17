# generate_all_routes_resumable.py (V7 - Definitive City-Wide Version)
import pandas as pd
import networkx as nx
from math import radians, sin, cos, sqrt, atan2
import os
import time
from sqlalchemy import create_engine, text
from config import DB_CONFIG

class CityWideRoutesGenerator:
    def __init__(self, gtfs_master_directory):
        self.gtfs_dir = gtfs_master_directory
        print("--- Step 1: Loading ALL Master GTFS files ---")
        self._load_and_clean_data()
        print("Master data loaded and cleaned successfully.")
        
        print("\n--- Step 2: Building the FULL City-Wide road network graph ---")
        self.G = nx.DiGraph()
        self._build_graph()
        print("\nFull city graph built successfully.")
        print(f"FINAL GRAPH: {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")
        if self.G.number_of_edges() > self.G.number_of_nodes() * 3: # Healthy check
             print("SUCCESS! The number of edges is high, indicating a well-connected city-wide graph.")
        else:
             print("WARNING: The number of edges seems low for a full city network.")

    def _load_and_clean_data(self):
        """Loads and cleans all data without any borough-specific filtering."""
        # We load all trips, and all the stops that those trips use.
        self.stop_times_df = pd.read_csv(os.path.join(self.gtfs_dir, 'master_stop_times.csv'), dtype=str)
        self.stops_df = pd.read_csv(os.path.join(self.gtfs_dir, 'master_stops.csv'), dtype=str)
        
        # Clean the data types
        self.stops_df.drop_duplicates(subset='stop_id', inplace=True)
        self.stops_df['stop_lat'] = pd.to_numeric(self.stops_df['stop_lat'], errors='coerce')
        self.stops_df['stop_lon'] = pd.to_numeric(self.stops_df['stop_lon'], errors='coerce')
        self.stops_df.dropna(subset=['stop_lat', 'stop_lon'], inplace=True)
        self.stop_times_df['stop_sequence'] = pd.to_numeric(self.stop_times_df['stop_sequence'], errors='coerce')
        self.stop_times_df.dropna(subset=['stop_sequence'], inplace=True)
        
        # Ensure stop_times only refers to stops that exist and have coordinates
        valid_stop_ids = set(self.stops_df['stop_id'])
        self.stop_times_df = self.stop_times_df[self.stop_times_df['stop_id'].isin(valid_stop_ids)]
        print(f"Loaded {len(self.stops_df)} unique stops and their corresponding time entries.")
        
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0; lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2]); dlon = lon2_rad - lon1_rad; dlat = lat2_rad - lat1_rad; a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2; c = 2 * atan2(sqrt(a), sqrt(1 - a)); return R * c

    def _build_graph(self):
        # Add all valid stops as nodes
        for stop_id in self.stops_df['stop_id']:
            self.G.add_node(stop_id)
            
        stops_coords = self.stops_df.set_index('stop_id')[['stop_lat', 'stop_lon']].to_dict('index')
        grouped_trips = self.stop_times_df.groupby('trip_id')
        
        for _, trip_data in grouped_trips:
            if len(trip_data) > 1:
                trip_data = trip_data.sort_values(by='stop_sequence')
                for i in range(len(trip_data) - 1):
                    from_stop_id = trip_data.iloc[i]['stop_id']
                    to_stop_id = trip_data.iloc[i+1]['stop_id']
                    # We can be confident both exist because of the cleaning step above
                    if from_stop_id in stops_coords and to_stop_id in stops_coords:
                        from_coords = stops_coords[from_stop_id]
                        to_coords = stops_coords[to_stop_id]
                        distance = self._calculate_distance(from_coords['stop_lat'], from_coords['stop_lon'], to_coords['stop_lat'], to_coords['stop_lon'])
                        self.G.add_edge(from_stop_id, to_stop_id, weight=distance)

    def generate_and_save_all_paths(self, table_name):
        print("\n--- Step 3: Calculating ALL City-Wide Paths (THIS WILL TAKE A VERY LONG TIME) ---")
        
        try:
            db_url = (f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}" f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
            engine = create_engine(db_url)
        except Exception as e:
            print(f"\nDATABASE ERROR: Could not create engine. Check config.py. Details: {e}"); return

        processed_start_ids = set()
        try:
            with engine.connect() as connection:
                if engine.dialect.has_table(connection, table_name):
                    result = connection.execute(text(f'SELECT DISTINCT start_id FROM {table_name};'))
                    processed_start_ids.update([row[0] for row in result])
                    print(f"Found {len(processed_start_ids)} already processed start_ids. Resuming...")
        except Exception as e:
            print(f"Could not check for existing table. Will create a new one. Details: {e}")
        
        # Process ALL nodes in the graph
        all_nodes_in_graph = list(self.G.nodes())
        nodes_to_calculate = [node for node in all_nodes_in_graph if node not in processed_start_ids]
        
        total_nodes = len(all_nodes_in_graph)
        print(f"Beginning path calculations for {len(nodes_to_calculate)} remaining start nodes...")
        
        start_time = time.time()
        
        for i, start_node in enumerate(nodes_to_calculate):
            paths_from_start_node = []
            try:
                paths = nx.single_source_dijkstra_path(self.G, source=start_node, weight='weight')
                lengths = nx.single_source_dijkstra_path_length(self.G, source=start_node, weight='weight')
            except (nx.NodeNotFound, KeyError): continue
            
            # Save paths to ALL other reachable nodes
            for end_node, path_list in paths.items():
                if start_node != end_node:
                    paths_from_start_node.append({'start_id': start_node, 'end_id': end_node, 'shortest_path_stops': '->'.join(map(str, path_list)), 'total_distance_km': lengths[end_node]})
            
            if paths_from_start_node:
                temp_df = pd.DataFrame(paths_from_start_node)
                try:
                    temp_df.to_sql(table_name, engine, if_exists='append', index=False, chunksize=5000)
                except Exception as e:
                    print(f"DB WRITE ERROR for start_node {start_node}: {e}"); continue
            
            if (i + 1) % 100 == 0:
                total_processed = len(processed_start_ids) + i + 1
                print(f"  Processed {total_processed}/{total_nodes} total start nodes... (runtime: {time.time() - start_time:.0f}s)")
        
        print(f"\n--- Path calculation phase complete in {time.time() - start_time:.2f} seconds ---")
        
        print("Creating database index for high-speed lookups...")
        try:
            with engine.connect() as connection:
                connection.execute(text(f'DROP INDEX IF EXISTS idx_{table_name}_routes;'))
                connection.execute(text(f'CREATE INDEX idx_{table_name}_routes ON {table_name} (start_id, end_id);'))
                connection.commit()
            print("Index created successfully.")
        except Exception as e:
            print(f"Could not create index. You may need to add it manually. Error: {e}")

        print("\n--- Process Complete ---")

# =============================================================================
if __name__ == "__main__":
    GTFS_FOLDER = 'master_data'
    # The new table name for our complete, city-wide knowledge base
    OUTPUT_TABLE_NAME = 'all_city_routes' 
    
    generator = CityWideRoutesGenerator(GTFS_FOLDER)
    generator.generate_and_save_all_paths(OUTPUT_TABLE_NAME)