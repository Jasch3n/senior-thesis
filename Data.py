from torch.utils.data import DataLoader, Dataset
import torch

class FuncData(Dataset):
    """
        Dataset object for 1D function data
    """
    def __init__(self, X, Y):
        """
        Initialize dataset variables.
        :param X: Features of shape [n], where n is number of samples.
        :param Y: Labels of shape [n], where n is number of samples.
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        """
        Returns the number of samples.
        :return: The number of samples.
        """
        return self.X.shape[0]

    def __getitem__(self, index):
        """
        Returns feature and label of the sample at the given index.
        :param index: Index of a sample.
        :return: Feature and label of the sample at the given index.
        """
        return torch.tensor(self.X[index]).float().unsqueeze(-1), torch.tensor([self.Y[index]]).float()
    
# TODO: implement this collate function for further utility!
def func_collate(items):
    pass