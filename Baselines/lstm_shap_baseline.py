import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
torch.manual_seed(42)

# Add PCMCI_robotic to path to import reporting_helper
sys.path.append(os.path.abspath("PCMCI/PCMCI_robotic"))
import reporting_helper

# Dataset logic
PREFIX = "PCMCI/PCMCI_robotic/pepper_csv/"
NORMAL_FILE = "normal.csv"
ATTACK_FILES = ["WheelsControl.csv", "JointControl.csv", "LedsControl.csv"]

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(h)
        return out

def create_sequences(data, seq_length=5):
    xs = []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
    return np.array(xs)

def main():
    print("\n========== PHASE 2: LSTM + SHAP BASELINE ==========")
    np.random.seed(42)
    torch.manual_seed(42)
    os.makedirs("Baselines/outputs", exist_ok=True)
    
    causal_path = os.path.join(PREFIX, "pepper_normal.npz")
    if not os.path.exists(causal_path):
        print("Causal model missing. Cannot match nonconst/subsample.")
        return
    
    f = np.load(causal_path, allow_pickle=True)
    subsample, nonconst = int(f["subsample"]), f["nonconst"]
    var_names = f["var"][nonconst].tolist()

    print("Loading Robotics normal data...")
    normal_df = pd.read_csv(os.path.join(PREFIX, NORMAL_FILE), delimiter=",")
    normal_data = np.nan_to_num(normal_df.drop(columns=["timestamp"]).values[::subsample, nonconst])
    
    scaler = StandardScaler()
    normal_data_scaled = scaler.fit_transform(normal_data)
    
    seq_length = 5
    X_train = create_sequences(normal_data_scaled, seq_length)
    
    model = LSTMAutoencoder(input_dim=len(nonconst), hidden_dim=16)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    dataset = TensorDataset(X_train_t)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    print("Training PyTorch LSTM Autoencoder (5 epochs)...")
    model.train()
    for epoch in range(5):
        for batch in dataloader:
            optimizer.zero_grad()
            out = model(batch[0])
            loss = criterion(out, batch[0])
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        train_out = model(X_train_t)
        normal_mse = torch.mean((train_out[:, -1, :] - X_train_t[:, -1, :])**2, dim=1).numpy()
    
    threshold = np.percentile(normal_mse, 95.0)
    print(f"95.0th Percentile Reconstruction Threshold: {threshold:.4f}")
    
    attack_scores_by_file = {}
    attack_labels_by_file = {}
    target_wise_residuals = {}
    
    for attack_file in ATTACK_FILES:
        print(f"Evaluating {attack_file}...")
        df = pd.read_csv(os.path.join(PREFIX, attack_file), delimiter=",")
        attack_data = np.nan_to_num(df.drop(columns=["timestamp"]).values[::subsample, nonconst])
        attack_data_scaled = scaler.transform(attack_data)
        X_attack = create_sequences(attack_data_scaled, seq_length)
        
        if len(X_attack) == 0: continue
            
        X_attack_t = torch.tensor(X_attack, dtype=torch.float32)
        with torch.no_grad():
            out = model(X_attack_t)
            mse = torch.mean((out[:, -1, :] - X_attack_t[:, -1, :])**2, dim=1).numpy()
            residuals = np.abs((out[:, -1, :] - X_attack_t[:, -1, :]).numpy())
        
        pad_len = seq_length
        mse_padded = np.concatenate([np.zeros(pad_len), mse])
        attack_scores_by_file[attack_file] = mse_padded
        attack_labels_by_file[attack_file] = np.ones(len(mse_padded))
        
        residuals_padded = np.vstack([np.zeros((pad_len, len(nonconst))), residuals])
        target_wise_residuals[attack_file] = residuals_padded

    normal_scores_padded = np.concatenate([np.zeros(seq_length), normal_mse])
    
    reporting_helper.generate_plots_and_reports(
        output_dir="Baselines/outputs",
        model_variant="LSTM Autoencoder",
        model_name="Neural Baseline",
        normal_scores=normal_scores_padded,
        attack_scores_by_file=attack_scores_by_file,
        threshold=threshold,
        var_names=var_names,
        target_indices=list(range(len(nonconst))),
        directed_edges=[],
        target_wise_residuals_by_file=target_wise_residuals,
        attack_labels_by_file=attack_labels_by_file,
        dataset_name="Robotics (Pepper)"
    )

    print("\n--- SHAP POST-HOC XAI ---")
    class MSEWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            out = self.model(x)
            return torch.mean((out[:, -1, :] - x[:, -1, :])**2, dim=1).unsqueeze(1)
    
    mse_model = MSEWrapper(model)
    background = X_train_t[:100]
    e = shap.DeepExplainer(mse_model, background)
    
    attack_file = ATTACK_FILES[0]
    df = pd.read_csv(os.path.join(PREFIX, attack_file), delimiter=",")
    attack_data_scaled = scaler.transform(np.nan_to_num(df.drop(columns=["timestamp"]).values[::subsample, nonconst]))
    X_test_t = torch.tensor(create_sequences(attack_data_scaled, seq_length), dtype=torch.float32)
    
    with torch.no_grad():
        mse_out = mse_model(X_test_t).squeeze().numpy()
        anomaly_idx = np.where(mse_out > threshold)[0]
    
    if len(anomaly_idx) > 0:
        sample_idx = anomaly_idx[:50]
        test_samples = X_test_t[sample_idx]
        
        # DeepExplainer can occasionally throw warnings, but this wrapper forces scalar outputs
        # Disable check_additivity because PyTorch LSTM gradients are an approximation in DeepExplainer
        shap_values = e.shap_values(test_samples, check_additivity=False)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        shap_values_2d = np.abs(shap_values).sum(axis=1)
        test_samples_2d = test_samples[:, -1, :].numpy()
        
        plt.figure()
        shap.summary_plot(shap_values_2d, test_samples_2d, feature_names=var_names, show=False)
        plt.savefig("Baselines/outputs/shap_feature_importance.png", bbox_inches="tight")
        print("Authentic SHAP summary plot saved to: Baselines/outputs/shap_feature_importance.png")
    else:
        print("No anomalies detected for SHAP.")

if __name__ == "__main__":
    main()
