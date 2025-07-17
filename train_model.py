# train_model.py (V2.2 - Definitive Stable Version)
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import time
import matplotlib.pyplot as plt
import copy
import os

# --- Dataset and Model Definition (Unchanged) ---
class TabularDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.targets[idx]

class RegressionNet(nn.Module):
    def __init__(self, input_size):
        super(RegressionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.network(x)

# --- Plotting Function (Unchanged) ---
def plot_and_save_history(history, best_epoch, output_path, version):
    print("\nGenerating performance chart...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    ax1.plot(history['val_loss'], label='Validation Loss (MSE)'); ax1.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch+1})'); ax1.set_title('Model Loss per Epoch'); ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss (MSE)'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['val_mae'], label='Validation MAE'); ax2.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch+1})'); ax2.set_title('Model MAE per Epoch'); ax2.set_xlabel('Epochs'); ax2.set_ylabel('Mean Absolute Error (s)'); ax2.legend(); ax2.grid(True)
    ax3.plot(history['val_r2'], label='Validation R-squared'); ax3.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch+1})'); ax3.set_title('Model R-squared per Epoch'); ax3.set_xlabel('Epochs'); ax3.set_ylabel('R-squared Score'); ax3.legend(); ax3.grid(True)
    plt.tight_layout(); chart_filename = os.path.join(output_path, f'bus_training_performance_{version}.png'); plt.savefig(chart_filename); plt.close()
    print(f"Training chart saved to '{chart_filename}'")


def train_pytorch_model_v2():
    print("--- PyTorch V2 Model Training (Stable Version) ---")
    
    VERSION = "v2"
    EPOCHS = 100
    BATCH_SIZE = 2048
    LEARNING_RATE = 0.0005 # Using a more conservative learning rate for stability
    PATIENCE = 10
    
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    historical_data_path = os.path.join(script_dir, 'historical_data', 'historical_training_data_v2.csv')
    model_output_path = os.path.join(script_dir, f'trained_model_pytorch_{VERSION}')

    if not os.path.exists(model_output_path): os.makedirs(model_output_path)

    print(f"Loading V2 data from '{historical_data_path}'...")
    df = pd.read_csv(historical_data_path)
    df = df.loc[df.groupby('trip_id').cumcount(ascending=False) != 0].copy()

    print("Preprocessing data using Label Encoding...")
    categorical_features = ['route_id', 'time_of_day']
    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    numeric_features = [
        'stop_sequence', 'current_delay', 'total_trip_distance',
        'stops_in_trip', 'progress_ratio'
    ]
    final_feature_columns = numeric_features + categorical_features
    target_column = 'final_delay'
    
    X = df[final_feature_columns].values
    y = df[target_column].values.reshape(-1, 1)

    #
    # --- THIS IS THE FIX ---
    # We apply the scaler to the ENTIRE feature matrix X after all encoding is done.
    # This ensures that our large, label-encoded integers are properly scaled.
    print("Scaling all features for network stability...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # --- END OF FIX ---

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_size = X_train.shape[1]
    model = RegressionNet(input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()

    # --- Training Loop (Unchanged) ---
    epochs_no_improve = 0; best_val_loss = float('inf'); best_model_state = None; best_epoch = 0
    history = {'val_loss': [], 'val_mae': [], 'val_r2': []}
    print(f"Starting training for up to {EPOCHS} epochs...")
    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device); optimizer.zero_grad(); outputs = model(features)
            loss = criterion(outputs, targets); loss.backward(); optimizer.step()
        model.eval(); val_mse_loss_sum, val_mae_sum = 0.0, 0.0; all_targets, all_predictions = [], []
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device); outputs = model(features)
                val_mse_loss_sum += criterion(outputs, targets).item() * features.size(0)
                val_mae_sum += torch.abs(outputs - targets).sum().item(); all_targets.extend(targets.cpu().numpy()); all_predictions.extend(outputs.cpu().numpy())
        avg_val_mse = val_mse_loss_sum / len(val_dataset); avg_val_mae = val_mae_sum / len(val_dataset); val_r2 = r2_score(all_targets, all_predictions)
        history['val_loss'].append(avg_val_mse); history['val_mae'].append(avg_val_mae); history['val_r2'].append(val_r2)
        print(f"Epoch {epoch+1}/{EPOCHS}.. Val Loss: {avg_val_mse:.2f}.. Val MAE: {avg_val_mae:.2f}s.. Val R²: {val_r2:.4f}")
        scheduler.step(avg_val_mse)
        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse; best_epoch = epoch; epochs_no_improve = 0; best_model_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best model found at epoch {epoch+1}!")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {PATIENCE} epochs with no improvement."); break
    end_time = time.time()
    print(f"\nTraining complete in {end_time - start_time:.2f} seconds.")

    # --- Save Best Model and Supporting Files ---
    print(f"Loading model from best epoch ({best_epoch+1}) to save.")
    model.load_state_dict(best_model_state)
    
    model_filename = os.path.join(model_output_path, f'bus_delay_predictor_{VERSION}.pth')
    scaler_filename = os.path.join(model_output_path, f'bus_scaler_{VERSION}.joblib')
    encoders_filename = os.path.join(model_output_path, f'bus_encoders_{VERSION}.joblib')

    torch.save(model.state_dict(), model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(encoders, encoders_filename)
    
    print(f"\n- PyTorch model state saved to '{model_filename}'")
    print(f"- Scaler saved to '{scaler_filename}'")
    print(f"- Encoders saved to '{encoders_filename}'")
    
    plot_and_save_history(history, best_epoch, model_output_path, VERSION)
    print(f"\n--- PyTorch Model Training Process ({VERSION.upper()}) Complete ---")

if __name__ == "__main__":
    train_pytorch_model_v2()