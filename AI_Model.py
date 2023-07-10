import numpy as np
import pickle
import torch

from time import time, sleep
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

raycastData = pickle.load(open("raycast_values", "rb"))
targets = pickle.load(open("solutions", "rb"))

class DrivingDataset(Dataset):
    def __init__(self, transform=None):
        self.data = pickle.load(open("raycast_values", "rb"))
        self.target = torch.LongTensor(pickle.load(open("solutions", "rb")))

        self.data = torch.Tensor(self.data)
        self.target = torch.Tensor(self.target)

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def create_model():
    input_size = 13
    hidden_sizes = [128, 64]
    output_size = 3

    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                        nn.Sigmoid(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.Sigmoid(),
                        nn.Linear(hidden_sizes[1], output_size))

    return model

def load_model(device):
    model = create_model().to(device)
    model.load_state_dict(torch.load('./racing_AI_model.pt'))
    model.eval()

    return model


def train(device):
    dataloader = DataLoader(DrivingDataset(), batch_size=128, shuffle=True)

    model = create_model().to(device)
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    startTime = time()
    epochs = 30

    for e in range (0, epochs):
        runningLoss = 0

        for raycastValues, target in dataloader:
            batchData = raycastValues.to(torch.float32).to(device)
            batchTargets = target.to(torch.float32).to(device)

            optimizer.zero_grad()

            batchOutput = model(batchData)

            loss = nn.MSELoss()(batchOutput, batchTargets)
            loss.backward()
            optimizer.step()
            
            runningLoss += loss.item()

        print("Epoch: ", e, " Training Loss: ", runningLoss/len(dataloader))

    print("Training Time: ", (time() - startTime)/60)

    torch.save(model.state_dict(), './racing_AI_model.pt') 

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(device)

if __name__ == "__main__":
    main()