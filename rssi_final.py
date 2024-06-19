import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.utils.data


class EarlyStopping:
    def __init__(self, patience=100, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.patience_count = 0
        self.best_loss = float('inf')
        self.stop_training = False

    def __call__(self, test_loss):
        if test_loss > self.best_loss - self.min_delta:
            self.patience_count += 1
            if self.patience_count >= self.patience:
                self.stop_training = True
        else:
            self.best_loss = test_loss
            self.patience_count = 0


# +
dff_train = pd.read_csv('/workspace/RSSI_prediction/data/dff_treino_normalizado.csv', delimiter=',')
dff_val = pd.read_csv('/workspace/RSSI_prediction/data/dff_validacao_normalizado.csv', delimiter=',')
dff_test = pd.read_csv('/workspace/RSSI_prediction/data/dff_teste_normalizado.csv', delimiter=',')

X_train = dff_train[['speed_vx', 'speed_vy', 'speed_vz', 'battery_percent', 'angle_phi', 'angle_psi', 'angle_theta', 'gps_amsl_altitude', 'landcover', 'distance_to_base']].to_numpy()
y_train = dff_train['wifi_signal_mW'].to_numpy()
X_val = dff_val[['speed_vx', 'speed_vy', 'speed_vz', 'battery_percent', 'angle_phi', 'angle_psi', 'angle_theta', 'gps_amsl_altitude', 'landcover', 'distance_to_base']].to_numpy()
y_val = dff_val['wifi_signal_mW'].to_numpy()
X_test = dff_test[['speed_vx', 'speed_vy', 'speed_vz', 'battery_percent', 'angle_phi', 'angle_psi', 'angle_theta', 'gps_amsl_altitude', 'landcover', 'distance_to_base']].to_numpy()
y_test = dff_test['wifi_signal_mW'].to_numpy()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

print("Forma do tensor X_train:", X_train_tensor.shape)
print("Forma do tensor y_train:", y_train_tensor.shape)
print("Forma do tensor X_val:", X_val_tensor.shape)
print("Forma do tensor y_val:", y_val_tensor.shape)
print("Forma do tensor X_test:", X_test_tensor.shape)
print("Forma do tensor y_test:", y_test_tensor.shape)


# -

def init_weights(model):
    for _, module in model.named_children():
        if hasattr(module, 'weight'):
            module.weight.data = torch.randn(module.weight.data.shape)


# +
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(10, 10)  
        self.sigmoid = nn.Sigmoid()   
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))  
        x = self.fc3(x)
        return x


model = MLP().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# +
epochs = 500
batch_size = 64
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

train_losses, val_losses, test_losses, mae_losses, mse_losses, rmse_losses, r2_scores = [], [], [], [], [], [], []
early_stopping = EarlyStopping(patience=100, min_delta=0.001)

weights_list = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    test_loss, mae_sum, mse_sum, total_sum = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            
            mae_sum += nn.L1Loss(reduction='sum')(outputs, targets).item()
            mse = nn.MSELoss(reduction='sum')(outputs, targets).item()
            mse_sum += mse
            total_sum += (targets - torch.mean(y_test_tensor)).pow(2).sum().item()

    test_loss /= len(test_loader.dataset)
    mae = mae_sum / len(test_loader.dataset)
    mse = mse_sum / len(test_loader.dataset)
    rmse = torch.sqrt(torch.tensor(mse))
    r2 = 1 - (mse_sum / total_sum)

    test_losses.append(test_loss)
    mae_losses.append(mae)
    mse_losses.append(mse)
    rmse_losses.append(rmse.item())
    r2_scores.append(r2)

    print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')
    early_stopping(val_loss)
    if early_stopping.stop_training:
        print(f'Stopping early at epoch {epoch+1}')
        break
    
    weights_list = []

for name, param in model.named_parameters():
    weights_list.append((name, param.detach().cpu()))

model.eval()

results_df = pd.DataFrame({
    'Epoch': range(1, len(train_losses) + 1),
    'Train Loss': train_losses,
    'Validation Loss': val_losses,
    'Test Loss': test_losses,
    'MAE': mae_losses,
    'MSE': mse_losses,
    'RMSE': rmse_losses,
    'R²': r2_scores
})

results_dir = '/workspace/RSSI_prediction/results_2'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

results_df.to_csv(os.path.join(results_dir, 'final_results26.csv'), index=False)

weights_path = os.path.join(results_dir, 'final_weights26.pth')
torch.save(dict(weights_list), weights_path)

torch.save(model, os.path.join(results_dir, 'model22.pth'))
