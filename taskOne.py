# Karan Shah
# Build and train a network to recognize digits 

# Import libraries 
import sys
import os
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Class Definitions
class MyNetwork(nn.Module):
    
    # Initialize network structure
    def __init__(self):
        super().__init__()
        self.cvstack = nn.Sequential(
                            nn.Conv2d(1, 10, kernel_size=5),
                            nn.MaxPool2d(2),
                            nn.ReLU(),
                            nn.Conv2d(10, 20, kernel_size=5),
                            nn.Dropout(0.5),
                            nn.MaxPool2d(2),
                            nn.ReLU(),
                            nn.Flatten(), 
                            nn.Linear(4 * 4 * 20, 50),
                            nn.ReLU(),
                            nn.Linear(50, 10)
                        )
        
    # Computes a forward pass for the network
    def forward(self, x):
        x = self.cvstack(x)
        predLobProb = F.log_softmax(x, dim=1) #dim??
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

# Plots the data
def plotData(images, labels):
    # Create 8 by 8 figure
    canvas = plt.figure(figsize=(8, 8))

    cols, rows = 3, 2
    
    # Add 6 gray scale images and add to canvas
    for i in range(cols * rows):
        canvas.add_subplot(rows, cols, i + 1)
        
        img, label = images[i], labels[i]
        
        plt.title(label=label.item())
        canvas.add_subplot(rows, cols, i + 1)
        plt.axis("off")
        plt.tight_layout()
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

# Plots the results
def plotResults(trainCounter, trainLosses, testCounter, testLosses):
    # Create 8 by 8 figure
    canvas = plt.figure(figsize=(8, 8))
    # Plots the losses for train and test
    plt.plot(trainCounter, trainLosses, color = 'blue')
    plt.scatter(testCounter, testLosses, color = 'red')
    plt.legend(['Train Loss', 'Test Loss'], loc = 'upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.show()

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
    test_losses.append(test_loss)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

# Method to train/test the network and plot the results
def train_network(model, trainDataLoader, testDataLoader, epochs, optimizer, lossFunction):
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(trainDataLoader.dataset) for i in range(0, epochs + 1)]


    test(testDataLoader, model, lossFunction, test_losses)
    # Iterate over epochs
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        print()
        print("-----------------------")
        train(epoch, trainDataLoader, model, lossFunction, optimizer, train_losses, train_counter)
        test(testDataLoader, model, lossFunction, test_losses)

    # Plot the results    
    plotResults(train_counter, train_losses, test_counter, test_losses)

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
    numEpochs = 5

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
    
    # Print diagram of network
    model = MyNetwork()
    # print(model)

    # # Setting the loss function and optimizer for network
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)

    # # Fix image for test losses
    train_network(model, train_dataloader, test_dataloader, epochs=numEpochs, optimizer=optimizer, lossFunction=loss_fn)

    # # Save the model.
    os.makedirs("models", exist_ok = True)
    torch.save(model, "models/model.pth")
    print("Saved PyTorch Model State to model.pth")

    return

if __name__ == "__main__":
    main(sys.argv)