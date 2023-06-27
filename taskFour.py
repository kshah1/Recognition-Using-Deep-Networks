# Karan Shah
# Hyperparameter tuning the network. 

# Import libraries
import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib
from itertools import product
import numpy as np


# Class Definitions
class MyNetwork(nn.Module):
    
    # Initialize network structure
    def __init__(self, channel, kernelSize, inputChannels, inputHeight, inputWidth):
        super().__init__()
        
        # self.firstConvLayerDim = inputHeight - kernelSize + 1
        # self.firstMaxPool2dDim = self.firstConvLayerDim // 2
        # self.secondConvLayerDim = self.firstMaxPool2dDim - kernelSize + 1
        # self.secondMaxPool2dDim = self.secondConvLayerDim // 2
        # self.flattenedDim = ( channel * 2 ) * (self.secondMaxPool2dDim ** 2)
        
        self.cvstack = nn.Sequential(
                            nn.Conv2d(1, channel, kernel_size = kernelSize),
                            nn.MaxPool2d(2),
                            nn.ReLU(),
                            nn.Conv2d(channel, channel * 2, kernel_size = kernelSize),
                            nn.Dropout(0.5),
                            nn.MaxPool2d(2),
                            nn.ReLU(),
                            nn.Flatten()
                        )
        
        # Dummy input to dynamically obtain the flattened dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, inputChannels, inputHeight, inputWidth)
            flattenedDim = self.cvstack(dummy_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flattenedDim, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        
    # Computes a forward pass for the network
    def forward(self, x):
        x = self.cvstack(x)
        x = self.fc(x)
        predLobProb = F.log_softmax(x, dim=1)
        return predLobProb

# Helper Functions

# Loads the data
def loadData():
    trainingData = datasets.MNIST(
                                root="data",
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                          (0.1307,), (0.3081,))
                                ])
                   )
    
    testData = datasets.MNIST(
                            root="data",
                            train=False,
                            download=True,
                            transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                      (0.1307,), (0.3081,))
                            ])
                    )
    
    return trainingData, testData

# Method to train the model 
def train(epoch, dataloader, model, loss_fn, optimizer, train_losses, train_counter):
    # Size of dataset
    size = len(dataloader.dataset)
    print(f"Size of Train Data: {size}")
    # Set model in train mode
    model.train()
    
    # Iterate through the data
    # X is shape 64, 1, 28, 28 with default batch settings
    # y is shape 64, 1 with default batch settings
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Prints every 10 batches (ex. 938 batches for train data of 60k images of batch 64. But only prints up to 930 batches)
        if batch % 10 == 0:
            print("Batch: {}".format(batch))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
            epoch, (batch + 1) * len(X), size,
            100. * ((batch + 1) * len(X)) / size, loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
            (batch * len(X)) + (epoch * size))

# Method to test the model            
def test(dataloader, model, loss_fn, test_losses):
    # Size of dataset
    size = len(dataloader.dataset)
    print(f"Size of Test Data: {size}")
    # Number of batches
    numBatches = len(dataloader)
    # Set model to evaluation mode
    model.eval()
    test_loss, correct = 0, 0
    # No gradient calculations during testing
    with torch.no_grad():
        # X is 1000, 1, 28, 28 with default batch settings
        # y is 1000, 10 
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= numBatches
    # test_losses.append(test_loss)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    return test_loss

# Method to train/test the network
def train_network(model, trainDataLoader, testDataLoader, epochs, optimizer, lossFunction):
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(trainDataLoader.dataset) for i in range(epochs)]
    
    lossResultsOverEpochs = []

    # Iterate over epochs
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        print()
        print("-----------------------")
        train(epoch, trainDataLoader, model, lossFunction, optimizer, train_losses, train_counter)
        lossResult = test(testDataLoader, model, lossFunction, test_losses)
        lossResultsOverEpochs.append(lossResult)

    return lossResultsOverEpochs


# Main function 
def main(argv):
    # Seeds used for repeatability
    random_seed = 42
    torch.backends.cudnn.enabled = False # type: ignore
    torch.manual_seed(random_seed)

    # Main variables
    batchSizeTrain = 64
    batchSizeTest = 1000
    learningRate = 0.01
    momentum = 0.5

    # Modifications to network
    sizeOfFilters = [2, 3, 5, 7]
    numberOfFilters = [5, 10, 15, 20]
    # sizeOfFilters = [2, 3, 5, 10] 
    # numberOfFilters = [5, 10, 15, 20] 
    numEpochs = [2, 5, 7]
    # numEpochs = [5, 10, 15] # Add this in after

    cartesianProduct = list(product(sizeOfFilters, numberOfFilters, numEpochs))

    # Load datasets
    trainingData, testData = loadData()

    # Wrap the datasets
    train_dataloader = DataLoader(trainingData, batch_size=batchSizeTrain)
    
    test_dataloader = DataLoader(testData, batch_size=batchSizeTest)

    # Iterate and plot first 6 examples
    # Uncomment to visualize
    # for X, y in train_dataloader:
    #     print(f"Shape of X [N, C, H, W]: {X.shape}")
    #     print(f"Shape of y: {y.shape} {y.dtype}")
    #     plotData(X, y)
    #     break

    losses = []
    # Print diagram of network
    for sizeOfFilters, channel, numEpochs in cartesianProduct:
        model = MyNetwork(channel, sizeOfFilters, 1, 28, 28)
        print(model)

        # Setting the loss function and optimizer for network
        loss_fn = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)

        # Fix image for test losses
        lossesOverEpochs = train_network(model, train_dataloader, test_dataloader, epochs=numEpochs, optimizer = optimizer, lossFunction = loss_fn)
        losses.append(lossesOverEpochs[-1])
    
    # Extract x, y, and epoch values from data
    x = ["({}, {})".format(filterSize, channel) for filterSize, channel, _ in cartesianProduct]
    epochs = [e for _, _, e in cartesianProduct]
    y = losses

    print(x)
    print(epochs)
    print(y)

    # Extract unique x-axis values
    unique_x = list(set(x))
    unique_x.sort()

    # Define the width of each bar
    bar_width = 0.35

    # Create an array of indices for each x-value
    indices = np.arange(len(unique_x))

    # Create a bar plot for each epoch
    for epoch in set(epochs):
        filtered_x = [xi for xi, ei in zip(x, epochs) if ei == epoch]
        filtered_y = [yi for yi, ei in zip(y, epochs) if ei == epoch]
        plt.bar(indices, filtered_y, width=bar_width, label=f'Epoch {epoch}')

        # Update x-axis indices for the next epoch
        indices = [ind + bar_width for ind in indices]

    # Set labels and title
    plt.xlabel('(FS, FN)')
    plt.ylabel('Losses')
    plt.title('Hyperparam Tuning')

    # Set x-axis tick locations and labels
    plt.xticks(np.arange(len(unique_x)) + bar_width * (len(set(epochs)) / 2), unique_x)
    plt.xticks(rotation=45)

    # Add a legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)