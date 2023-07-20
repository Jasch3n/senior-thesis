from Models import MultiLayerNN, OneLayerNN
import torchinfo, torch, Data
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from IPython.display import display, clear_output
import time
from Utility import getFourierCoeffs

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
device = torch.device("cpu")

print("Using device", device)

def true_func(x):
    return np.sin(np.pi*x)

def func1(x):
    noise = 0.2*np.sin(4*np.pi*x) + 0.15*np.sin(7*np.pi*x)
    return true_func(x) + noise

def add_noise_func(x, p):
    vals = true_func(x)
    for i in range(len(vals)):
        if np.random.random() <= p:
            vals[i] = vals[i] + np.random.normal(loc=0, scale=0.25)
    return vals

np.random.seed(33)
SAMPLE_SIZE = 1200
x_train = np.linspace(-1,1,SAMPLE_SIZE)
y_train = func1(x_train)
xy_ds = Data.FuncData(x_train, y_train)
xy_dl = torch.utils.data.DataLoader(xy_ds, batch_size=100, shuffle=True)

def compute_NN_function(num_epochs, hidden_units, xy_dl):
    def getFourierCoeffs(x, func, num=11):
        num_pts = len(x)
        sin_coeffs = []
        cos_coeffs = []
        for i in range(1,num+1):
            sin_coeffs.append(sum(func(x) * np.sin(i*np.pi*x)) * ((np.max(x)-np.min(x))/num_pts))
            cos_coeffs.append(sum(func(x) * np.cos(i*np.pi*x)) * ((np.max(x)-np.min(x))/num_pts))
        zero_cos_coeff = sum(func(x)) * ((np.max(x)-np.min(x))/num_pts)
        return (sin_coeffs, [zero_cos_coeff] + cos_coeffs)

    LEARNING_RATE = 0.05
    np.random.seed(1113)
    torch.manual_seed(1113)
    model = OneLayerNN(hidden_size=hidden_units, activation=torch.nn.Tanh()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = torch.nn.MSELoss()

    sin_coeffs_mat = []
    nn_func = lambda x: model(torch.tensor(x).to(device).float().unsqueeze(-1)).cpu().detach().numpy().flatten()
    for epoch in range(num_epochs):
        sin_coeffs, _ = getFourierCoeffs(x_train, nn_func, num=12)
        sin_coeffs_mat.append(sin_coeffs)
        epoch_loss = 0
        for X,Y in xy_dl:
            _X = X.to(device)
            _Y = Y.to(device)
            Y_hat = model(_X)
            optimizer.zero_grad()
            L = loss(_Y, Y_hat)
            L.backward()
            optimizer.step()
            epoch_loss += float(L.item()) / len(xy_dl.dataset)
        print('Epoch',epoch,'| Loss:', epoch_loss, end='\r')

    return nn_func, sin_coeffs_mat

import os
SAVE_PATH = ".\\results\\one-layer-vary-depth-test"
if not os.path.isdir(SAVE_PATH):
    os.system("mkdir -p " + SAVE_PATH)

hidden_sizes = [16, 32, 48, 64, 85, 128, 256, 512, 1024, 2048, 4096, 5096, 6096, 7096, 8096, 9000, 10000]
print("Carrying out trial on hidden sizes:", hidden_sizes)
test_errors = []
sin_coeffs_mats = []
y_hat_mat = np.zeros((len(hidden_sizes), len(x_train)))
test_error_mat = np.zeros((len(hidden_sizes), 1))
NUM_EPOCHS = 1
for i in tqdm(range(len(hidden_sizes))):
    s = hidden_sizes[i]
    nn_func, sin_coeffs_mat = compute_NN_function(NUM_EPOCHS, s, xy_dl)
    y_hat = nn_func(x_train)
    y_hat_mat[i,:] = y_hat
    test_error = np.mean(np.square(y_hat - y_train))
    test_error_mat[i] = test_error
    test_errors.append(test_error)
    sin_coeffs_mats.append(sin_coeffs_mat)
sin_coeffs_mats = np.array(sin_coeffs_mats)

np.save(os.path.join(SAVE_PATH, "hidden-sizes"), hidden_sizes)
np.save(os.path.join(SAVE_PATH, "y-hat-mat"), y_hat_mat)
np.save(os.path.join(SAVE_PATH, "test-errors"), test_error_mat)