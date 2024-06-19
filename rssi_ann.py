import pandas as pd
import torch
import os
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam, SGD, NAdam
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from skorch.callbacks import Callback

# +
dff_train = pd.read_csv('/workspace/RSSI_prediction/data/dff_treino_normalizado.csv', delimiter=',')
dff_test = pd.read_csv('/workspace/RSSI_prediction/data/dff_teste_normalizado.csv', delimiter=',')
dff_val = pd.read_csv('/workspace/RSSI_prediction/data/dff_validacao_normalizado.csv', delimiter=',')

X_train = dff_train[['speed_vx', 'speed_vy', 'speed_vz', 'battery_percent', 'angle_phi', 'angle_psi', 'angle_theta', 'gps_amsl_altitude', 'landcover', 'distance_to_base']].to_numpy()
y_train = dff_train['wifi_signal_mW'].to_numpy()

X_test = dff_test[['speed_vx', 'speed_vy', 'speed_vz', 'battery_percent', 'angle_phi', 'angle_psi', 'angle_theta', 'gps_amsl_altitude', 'landcover', 'distance_to_base']].to_numpy()
y_test = dff_test['wifi_signal_mW'].to_numpy()

X_val = dff_val[['speed_vx', 'speed_vy', 'speed_vz', 'battery_percent', 'angle_phi', 'angle_psi', 'angle_theta', 'gps_amsl_altitude', 'landcover', 'distance_to_base']].to_numpy()
y_val = dff_val['wifi_signal_mW'].to_numpy()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

print("Forma do tensor X_train:", X_train_tensor.shape)
print("Forma do tensor y_train:", y_train_tensor.shape)
print("Forma do tensor X_test:", X_test_tensor.shape)
print("Forma do tensor y_test:", y_test_tensor.shape)
print("Forma do tensor X_val:", X_val_tensor.shape)
print("Forma do tensor y_val:", y_val_tensor.shape)


# +
class EarlyStoppingNoImprovement(Callback):
    def __init__(self, patience=100, best_loss=float('inf'), epochs_no_improvement=0, best_epoch=0, results={}):
        self.patience = patience
        self.epochs_no_improvement = epochs_no_improvement
        self.best_epoch = best_epoch        
        self.best_loss = best_loss
        self.results = results

    def on_epoch_end(self, net, **kwargs):
        current_loss = net.history[-1, 'train_loss']
        
        if current_loss < self.best_loss:
            self.best_epoch = len(net.history)
            self.best_loss = current_loss
            self.epochs_no_improvement = 0
        else:
            self.epochs_no_improvement += 1

        if self.epochs_no_improvement >= self.patience:
            raise KeyboardInterrupt

    def on_train_end(self, net, **kwargs):
        print(f"Final Epoch: {len(net.history)}")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best Loss: {self.best_loss}")

        self.results['final_epoch'].extend([len(net.history)])
        self.results['best_epoch'].extend([self.best_epoch])
        self.results['best_loss'].extend([self.best_loss])
        self.results['patience'].extend([self.patience])
        self.results['module'].extend([str(net.module_.__class__.__name__)])
        self.results['module__num_units'].extend([str(net.module_.num_units)])
        self.results['module__activation'].extend([str(net.module_.activation.__class__.__name__)])
        self.results['optimizer'].extend([str(net.optimizer_.__class__.__name__)])
        self.results['lr'].extend([net.optimizer_.param_groups[0]['lr']])
        
        results_df = pd.DataFrame(self.results)
        results_file = '/workspace/RSSI_prediction/results/results_ann_partial.csv'
        
        if os.path.exists(results_file):
            previous_results = pd.read_csv(results_file)
            combined_results = pd.concat([previous_results, results_df], axis=0)
        else:
            combined_results = results_df
        
        combined_results.to_csv(results_file, index=False)

class SimpleNeuralNet(nn.Module):
    def __init__(self, num_units=20, activation='relu'):
        super(SimpleNeuralNet, self).__init__()
        self.num_units = num_units
        self.dense1 = nn.Linear(10, num_units)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError("Unsupported activation function")
        self.dense2 = nn.Linear(num_units, 1)

    def forward(self, X):
        X = self.dense1(X)
        X = self.activation(X)
        X = self.dense2(X)
        return X

results = {
    'best_epoch': [],
    'best_loss': [],
    'final_epoch': [],
    'patience': [],
    'module': [],                
    'module__num_units': [],
    'module__activation': [],
    'optimizer': [],
    'lr': [],
}

net = NeuralNetRegressor(
    module=SimpleNeuralNet,
    max_epochs=500,
    lr=0.01,
    optimizer=torch.optim.SGD,
    criterion=nn.MSELoss(),
    iterator_train__shuffle=True,
    device=device,
    callbacks=[
        EarlyStoppingNoImprovement(
            patience=100,
            results=results
        ),
    ],
)

params = {
    'module__num_units': list(range(10, 41, 5)),
    'lr': [0.1, 0.01, 0.001, 0.0001],
    'optimizer': [torch.optim.Adam, torch.optim.SGD, torch.optim.NAdam],
    'module__activation': ['relu', 'tanh', 'sigmoid', 'elu'],
}

gs = GridSearchCV(net, params, refit=True, cv=3, verbose=3, scoring='neg_mean_squared_error')
gs.fit(X_train_tensor.cpu().numpy(), y_train_tensor.cpu().numpy())

print("Melhores parâmetros:", gs.best_params_)

y_pred = torch.from_numpy(gs.predict(X_test_tensor.cpu().numpy())).to(device)
mse_test = torch.mean((y_test_tensor.view(-1, 1) - y_pred) ** 2)
print("MSE no conjunto de teste:", mse_test.item())

y_pred_val = torch.from_numpy(gs.predict(X_val_tensor.cpu().numpy())).to(device)
mse_val = torch.mean((y_val_tensor.view(-1, 1) - y_pred_val) ** 2)
print("MSE no conjunto de validação:", mse_val.item())

df_best_params = pd.DataFrame([gs.best_params_])
df_best_params['MSE_test_set'] = mse_test.item()
df_best_params['MSE_val_set'] = mse_val.item()

df_best_params.to_csv('/workspace/RSSI_prediction/results/results_ann_final_final.csv', index=False)

# Salvar resultados parciais
results_partial = {
    'best_epoch': [gs.best_estimator_.history[-1]['epoch']],
    'best_loss': [gs.best_estimator_.history[-1]['train_loss']],
    'final_epoch': [len(gs.best_estimator_.history)],
    'patience': [gs.best_estimator_.callbacks[0].patience],
    'module': [str(gs.best_estimator_.module_.__class__.__name__)],
    'module__num_units': [gs.best_params_['module__num_units']],
    'module__activation': [gs.best_params_['module__activation']],
    'optimizer': [str(gs.best_params_['optimizer'].__name__)],
    'lr': [gs.best_params_['lr']]
}

df_results_partial = pd.DataFrame(results_partial)
results_file = '/workspace/RSSI_prediction/results/results_ann_partial.csv'

if os.path.exists(results_file):
    previous_results = pd.read_csv(results_file)
    combined_results = pd.concat([previous_results, df_results_partial], ignore_index=True)
else:
    combined_results = df_results_partial

combined_results.to_csv(results_file, index=False)
