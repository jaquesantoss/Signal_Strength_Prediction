import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.utils.data

# +
model = torch.load('/workspace/RSSI_prediction/results_2/model.pth')

model.eval()

dff2_sampled_RM = pd.read_csv('/workspace/RSSI_prediction/data/dff2_sampled_RM.csv', delimiter=',')
dff9_sampled_RM = pd.read_csv('/workspace/RSSI_prediction/data/dff9_sampled_RM.csv', delimiter=',')
dff12_sampled_RM = pd.read_csv('/workspace/RSSI_prediction/data/dff12_sampled_RM.csv', delimiter=',')

X_new = dff2_sampled_RM[['speed_vx', 'speed_vy', 'speed_vz', 'battery_percent', 'angle_phi', 'angle_psi', 'angle_theta', 'gps_amsl_altitude', 'landcover', 'distance_to_base']].values
X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

with torch.no_grad():
    y_pred = model(X_new_tensor)

predictions = y_pred.numpy()

df_predictions = pd.DataFrame(predictions, columns=['prediction'])
df_predictions.to_csv('/workspace/RSSI_prediction/results_3/predictions_dff2.csv', index=False)
