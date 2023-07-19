import torch
from torch import nn
import numpy as np

from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from Utility import getFourierCoeffs
from Data import FuncData
from Models import OneLayerNN
from tqdm import tqdm

def train_and_extract(xx, obj_func, data=[],
                      lr=0.05, hidden_size=15, num_epochs=350, num_coeffs=10,
                      sample_size=3000, batch_size=100, shuffle=True):
    """
        Train a neural network with one hidden layer over the domain specified to fit
        a specified function, and extract the dynamics of the Fourier coefficients 
        over the epochs.

        Note that obj_func needs to be an actual Python function!!!
    """
    # ============================ DATA GENERATION ===============================
    # Generate Data from our defined target (objective) function
    # xx = np.linspace(-1, 1, 10000)
    # obj_func = lambda x: np.sin(np.pi*x) + np.sin(4*np.pi*x) + np.sin(7*np.pi*x) 
    # can also add in this following noise + 0.1*np.random.randn(len(x))
    sample_size = 3000
    x_diff = np.max(xx) - np.min(xx)
    # Uniformly sample data if not specified
    if len(data)==0:
        data = x_diff*np.random.uniform(size=sample_size) + np.min(xx)
    data=data.astype(np.float32)
    output = obj_func(data); output=output.astype(np.float32)
    # Remember to wrap the training data in the PyTorch Dataloader object
    train_dataloader = DataLoader(FuncData(data, output), batch_size=batch_size, shuffle=shuffle)

    # ============================== MODEL DEFINITION ============================
    # Train the model on the generated data using Adam and MSE
    model = OneLayerNN(hidden_size=hidden_size)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # train(model, train_dataloader, loss_func, optimizer, num_epochs, print_info=False);
    sin_coeffs_mat = np.zeros((num_epochs, num_coeffs))
    cos_coeffs_mat = np.zeros((num_epochs, num_coeffs+1))
    # Use this anonymous function to extract the Fourier coefficients at every epoch
    func_hat = lambda x: model(torch.Tensor(x).unsqueeze(-1)).detach().numpy().flatten()

    # ============================== MODEL TRAINING ==============================
    # Initialize an empty list to save average losses in all epochs.
    _loss = []
    # Tell the model we are in training mode
    model.train()
    dataloader = train_dataloader
    # train network for num_epochs
    for epoch in tqdm(range(num_epochs), desc="Training ..."):
        sin_coeffs_mat[epoch], cos_coeffs_mat[epoch] = getFourierCoeffs(xx, func_hat, num=num_coeffs)
        # Initializing variables
        # Sum of losses in an epoch. Will be used to calculate average loss.
        epoch_loss_sum = 0
        # Iterate through all the data.
        for X,Y in dataloader:
            X = X.unsqueeze(-1)
            Yhat = model.forward(X)
            optimizer.zero_grad()
            loss = loss_func(Yhat, Y)
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * X.shape[0]

        # Append the average loss of the current epoch to list
        _loss.append(epoch_loss_sum / len(dataloader.dataset))

    return {
        "sin_coeffs_mat": sin_coeffs_mat,
        "cos_coeffs_mat": cos_coeffs_mat,
        "losses": _loss,
        "model": model
    }

def plot_coeffs(coeff_indx, coeffs_mat, target_coeffs=None, normalize=False):
    """
        Plots the growth of coefficients of different Fourier components over epochs
    """
    tmp_indx = np.array(coeff_indx) - 1
    plt.figure()
    for i in tmp_indx:
        if normalize: 
            plt.plot(coeffs_mat[:,i] / target_coeffs[i])
        else:
            plt.plot(coeffs_mat[:,i])
    # plt.title("Growth of Different Sine Fourier Modes during training")
    num_epochs, _ = coeffs_mat.shape
    if not target_coeffs==None and not normalize:
        plt.plot(np.arange(num_epochs), np.zeros(num_epochs)+target_coeffs)
    plt.legend([f"$c_{i}$" for i in coeff_indx])
    plt.xlabel("Epoch Number")
    plt.ylabel("Fourier Coefficient Magnitude")