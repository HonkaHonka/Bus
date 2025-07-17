# api.py (V4.6 - Definitive Imports)
import joblib, torch, pandas as pd, sqlite3, os
from sqlalchemy import create_engine, text
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles       # <-- THE CORRECT IMPORT LOCATION
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from config import DB_CONFIG

# --- 1. Pydantic Data Models (All necessary models are here) ---
class PredictionRequest(BaseModel): trip_id: str
class ETAEntry(BaseModel): stop_id: str; eta_seconds: int
class RoutePathDetails(BaseModel): full_path: List[str]; path_completed: List[str]; path_remaining: List[str]; total_distance_km: float
class PredictionResponse(BaseModel): trip_id: str; predicted_final_delay_seconds: float; current_status_from_db: Dict[str, Any]; static_route_details: RoutePathDetails; etas_to_future_stops: List[ETAEntry]
class RegulationResponse(PredictionResponse): regulation_advice: str; advice_level: str
class ActiveTrip(BaseModel): trip_id: str; route_id: str; current_delay: int; start_station: str; end_station: str
class Station(BaseModel): stop_id: str; stop_name: str
class TripForStation(BaseModel): trip_id: str; route_id: str; destination: str

# --- 2. LOAD V2 MODEL AND ASSETS (Unchanged) ---
print("--- Initializing Prediction API (V6.1 - Corrected) ---")
VERSION = "v2"; script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
model_path = os.path.join(script_dir, f'trained_model_pytorch_{VERSION}')
try:
    scaler = joblib.load(os.path.join(model_path, f'bus_scaler_{VERSION}.joblib')); encoders = joblib.load(os.path.join(model_path, f'bus_encoders_{VERSION}.joblib'))
    from train_model import RegressionNet
    input_size = scaler.n_features_in_; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionNet(input_size); model.load_state_dict(torch.load(os.path.join(model_path, f'bus_delay_predictor_{VERSION}.pth'), map_location=device, weights_only=True))
    model.to(device); model.eval(); print("V2 Model and assets loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load V2 model files. {e}"); model = None

# --- 3. CREATE DATABASE ENGINE & FASTAPI APP (Unchanged) ---
app = FastAPI(title="TransitPulse Web Application API (V6.1)")
try:
    db_url = (f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}" f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    engine = create_engine(db_url, pool_size=10, max_overflow=20); print("Database engine created successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not create database engine. {e}"); engine = None
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 4. REGULATION ENGINE & HELPER FUNCTIONS ---

# --- THIS IS THE RESTORED HELPER FUNCTION ---
async def get_full_prediction_data(trip_id: str):
    if engine is None: raise HTTPException(status_code=503, detail="Database connection is not available.")
    
    with engine.connect() as connection:
        live_data = connection.execute(text("SELECT * FROM live_status WHERE trip_id = :trip_id"), {"trip_id": trip_id}).mappings().first()
        if not live_data:
            raise HTTPException(status_code=404, detail=f"No live data for trip '{trip_id}'. Is the simulation running?")
        
        last_updated_time = live_data['last_updated']
        if (datetime.now(timezone.utc) - last_updated_time).total_seconds() > 30:
            raise HTTPException(status_code=408, detail=f"Stale data for trip '{trip_id}'. Last update was at {last_updated_time.isoformat()}.")
        
        # This now uses the clean columns from the live_status table
        start_id, end_id = live_data['start_id'], live_data['end_id']
        static_data = connection.execute(text("SELECT * FROM all_city_routes WHERE start_id = :start_id AND end_id = :end_id"), {"start_id": start_id, "end_id": end_id}).mappings().first()
        if not static_data:
            raise HTTPException(status_code=404, detail=f"No static route data found in knowledge base for trip from {start_id} to {end_id}.")

    # The AI feature creation logic remains the same
    full_path_list = static_data['shortest_path_stops'].split('->'); stops_in_trip = len(full_path_list)
    features = {
        'stop_sequence': live_data['stop_sequence'], 'current_delay': live_data['current_delay'],
        'total_trip_distance': static_data['total_distance_km'], 'stops_in_trip': stops_in_trip,
        'progress_ratio': (live_data['stop_sequence'] / stops_in_trip) if stops_in_trip > 0 else 0,
        'route_id': live_data['route_id'], 'time_of_day': live_data['time_of_day']
    }
    input_df = pd.DataFrame([features])
    for col, le in encoders.items():
        input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    final_feature_columns = ['stop_sequence', 'current_delay', 'total_trip_distance', 'stops_in_trip', 'progress_ratio'] + ['route_id', 'time_of_day']
    input_df = input_df[final_feature_columns]
    processed_features = scaler.transform(input_df)
    features_tensor = torch.tensor(processed_features, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = model(features_tensor)
    return dict(live_data), dict(static_data), prediction.item()

def get_regulation_advice(predicted_delay_seconds: float) -> (str, str):
    if predicted_delay_seconds > 300: return ("ACTION: Major delay predicted. Prioritize schedule. Minimize dwell time at next 5 stops.", "ACTION_REQUIRED")
    elif 90 < predicted_delay_seconds <= 300: return ("INFO: Minor delay predicted. Maintain pace and attempt to recover time safely.", "INFO")
    elif -120 <= predicted_delay_seconds <= 90: return ("INFO: On schedule. Maintain current pace.", "INFO")
    else: return ("ACTION: Running ahead of schedule. Hold at next major stop for 90 seconds to regulate timing.", "ACTION_REQUIRED")

# --- 5. API ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request): return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_passenger_dashboard(request: Request): return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/stations", response_class=HTMLResponse)
async def serve_stations_page(request: Request): return templates.TemplateResponse("stations.html", {"request": request})

@app.get("/driver", response_class=HTMLResponse)
async def serve_driver_dashboard(request: Request): return templates.TemplateResponse("driver.html", {"request": request})

@app.get("/api/active_trips", response_model=List[ActiveTrip])
async def api_get_active_trips():
    if engine is None: raise HTTPException(status_code=503, detail="Database connection is not available.")
    sql = text("""SELECT ls.trip_id, ls.route_id, ls.current_delay, s1.stop_name AS start_station, s2.stop_name AS end_station FROM live_status ls JOIN stops_directory s1 ON ls.start_id = s1.stop_id JOIN stops_directory s2 ON ls.end_id = s2.stop_id;""")
    with engine.connect() as connection:
        results = connection.execute(sql).mappings().all()
    return results

@app.get("/api/all_stations", response_model=List[Station])
async def api_get_all_stations():
    if engine is None: raise HTTPException(status_code=503, detail="Database connection is not available.")
    with engine.connect() as connection:
        query = text("SELECT stop_id, stop_name FROM stops_directory ORDER BY stop_name;")
        results = connection.execute(query).mappings().all()
    return results

@app.get("/api/trips_for_stop/{stop_id}", response_model=List[TripForStation])
async def api_get_trips_for_stop(stop_id: str):
    if engine is None: raise HTTPException(status_code=503, detail="Database connection is not available.")
    sql = text("""SELECT ls.trip_id, ls.route_id, end_stop.stop_name as destination FROM live_status ls JOIN all_city_routes acr ON (ls.start_id = acr.start_id AND ls.end_id = acr.end_id) JOIN stops_directory end_stop ON (acr.end_id = end_stop.stop_id) WHERE acr.shortest_path_stops LIKE :pattern""")
    search_pattern = f"%->{stop_id}%"
    with engine.connect() as connection:
        results = connection.execute(sql, {"pattern": search_pattern}).mappings().all()
    return results

@app.post("/api/predict", response_model=PredictionResponse)
async def api_predict_live(request: PredictionRequest):
    live_data, static_data, predicted_delay = await get_full_prediction_data(request.trip_id)
    full_path_list = static_data['shortest_path_stops'].split('->')
    etas_to_future_stops = []; cumulative_eta_seconds = 0; current_stop_index = live_data['stop_sequence'] - 1
    if current_stop_index < len(full_path_list) - 1:
        for i in range(current_stop_index + 1, len(full_path_list)):
            cumulative_eta_seconds += 120; etas_to_future_stops.append(ETAEntry(stop_id=full_path_list[i], eta_seconds=cumulative_eta_seconds))
    route_path_details = RoutePathDetails(full_path=full_path_list, path_completed=full_path_list[:current_stop_index + 1], path_remaining=full_path_list[current_stop_index + 1:], total_distance_km=static_data['total_distance_km'])
    return PredictionResponse(trip_id=request.trip_id, predicted_final_delay_seconds=predicted_delay, current_status_from_db=live_data, static_route_details=route_path_details, etas_to_future_stops=etas_to_future_stops)

@app.post("/api/regulation", response_model=RegulationResponse)
async def api_get_regulation(request: PredictionRequest):
    prediction_response = await api_predict_live(request)
    advice, level = get_regulation_advice(prediction_response.predicted_final_delay_seconds)
    return RegulationResponse(**prediction_response.dict(), regulation_advice=advice, advice_level=level)