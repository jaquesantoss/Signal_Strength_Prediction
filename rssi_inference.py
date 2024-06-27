#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(10, 35)
        self.relu = nn.ReLU()         
        self.fc3 = nn.Linear(35, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc3(x)             
        return x

model = torch.load('/workspace/RSSI_prediction/data/model23.pth', map_location=device)
model.to(device)
model.eval()

dff2_sampled_RM = pd.read_csv('/workspace/RSSI_prediction/data/dff2_sampled_RM_normalized.csv', delimiter=',')
dff9_sampled_RM = pd.read_csv('/workspace/RSSI_prediction/data/dff9_sampled_RM_normalized.csv', delimiter=',')
dff12_sampled_RM = pd.read_csv('/workspace/RSSI_prediction/data/dff12_sampled_RM_normalized.csv', delimiter=',')

X_new = dff2_sampled_RM[['speed_vx', 'speed_vy', 'speed_vz', 'battery_percent', 'angle_phi', 'angle_psi', 'angle_theta', 'gps_amsl_altitude', 'landcover', 'distance_to_base']].values
X_new_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred = model(X_new_tensor)

predictions = y_pred.cpu().numpy()
df_predictions = pd.DataFrame(predictions, columns=['prediction'])
df_predictions.to_csv('/workspace/RSSI_prediction/results_3/predictions_dff2_model23.csv', index=False)
