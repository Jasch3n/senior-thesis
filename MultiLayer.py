from Models import MultiLayerNN, OneLayerNN
import torchinfo, torch, Data
import numpy as np
import matplotlib.pyplot as plt 

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device", device)

def func1(x):
    return np.sin(np.pi*x) + np.sin(3*np.pi*x) #+ np.sin(8*np.pi*x)

x_data = np.linspace(-1,1,300).astype(np.float32)
y_data = func1(x_data).astype(np.float32)
xy_ds = Data.FuncData(x_data, y_data)
xy_dl = torch.utils.data.DataLoader(xy_ds, batch_size=50, shuffle=True)

# model = MultiLayerNN([32,32]).to(device)
# NOTE: What is the effect of increasing the number of layers 
# on the frequency components of the learned function?
model = OneLayerNN(hidden_size=1000).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
loss = torch.nn.MSELoss()
NUM_EPOCH = 200
torch.cuda.empty_cache()
for epoch in range(NUM_EPOCH):
    epoch_avg_loss = 0
    for X,Y in xy_dl:
        _X = X.unsqueeze(-1).to(device)
        _Y = Y.to(device)
        Y_hat = model(_X)
        optimizer.zero_grad()
        L = loss(_Y, Y_hat)
        L.backward()
        optimizer.step()
        epoch_avg_loss += float(L.item()) / len(xy_ds)
    print(epoch_avg_loss)

x_test = torch.tensor(2*np.random.random(size=50) - 1).float().unsqueeze(-1).to(device)
y_test = model(x_test).cpu().detach().numpy()
x_test = x_test.cpu().detach().numpy()
plt.scatter(x_test, y_test)
plt.scatter(x_test, func1(x_test))
plt.show()