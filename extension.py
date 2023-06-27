# Karan Shah
# Perform learning on extended greek letters

# Load the libraries
import sys
import torch
import torchvision
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from taskOne import MyNetwork

# Greek Data Set Transform
class GreekTransform:
    def __init__(self):
        pass
    
    # Grayscale, scale, crop, and invert the image
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x ) # type: ignore
        # Scale 133 * 36/128 = 27
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 ) # type: ignore
        # Crop to 28 28
        x = torchvision.transforms.functional.center_crop( x, (28, 28) ) # type: ignore
        return torchvision.transforms.functional.invert( x ) # type: ignore
    

# Helper Functions

# Load, greek transform, and normalize the data
def loadData(datapath, batchSize):
    return torch.utils.data.DataLoader(torchvision.datasets.ImageFolder( datapath, # type: ignore
                                                                            transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                                                         GreekTransform(),
                                                                                                                         torchvision.transforms.Normalize(
                                                                                                                         (0.1307,), (0.3081,) ) ] )),
                                                                                                                         batch_size = batchSize,
                                                                                                                         shuffle = True )
# Plot the train transfer results
def plotTrainTransferResults(trainCounter, trainLosses):
    # Create 8 by 8 figure
    canvas = plt.figure(figsize=(8, 8))
    plt.plot(trainCounter, trainLosses, color='blue')
#     plt.scatter(testCounter, testLosses, color='red')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.show()
    return

# Transfer learning train method for the model on greek letters
def trainTransfer(epoch, dataloader, model, loss_fn, optimizer, train_losses, train_counter):
    # Size of train dataset is 27
    size = len(dataloader.dataset)
    print(f"Size of Train Data: {size}")
    # Run model in train mode
    model.train()
    # X in this example is 5, 1, 28, 28 until 6th final batch
    # y is 5, 1 until 6th final batch
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 6 batches. Batches 0 to 4 consist 5 training examples so mod by 2
        if batch % 2 == 0:
            print("Batch: {}".format(batch))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
            epoch, (batch + 1) * len(X), size,
            100. * ((batch + 1) * len(X)) / size, loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
            (batch * len(X)) + (epoch * size))
    return 

# Train the network over number of epochs and plot the training error            
def train_network_transfer(model, trainDataLoader, epochs, optimizer, lossFunction):
    # Store train losses and train counter for plotting
    train_losses = []
    train_counter = []
    # Iterate over the epochs
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        print("-----------------------")
        trainTransfer(epoch, trainDataLoader, model, lossFunction, optimizer, train_losses, train_counter)
    # Plot the training error    
    plotTrainTransferResults(train_counter, train_losses)
    return
    
# Test on new greek letters
def testNewGreekLetters(model, testDataLoader):
    preds = []
    images = []
    labels = []

    # Set to evaluation mode
    model.eval()

    # Store the predictions, images, and labels
    with torch.no_grad():
        for batch, (X, y) in enumerate(testDataLoader):
            probPred = model(X)
            preds.append(probPred[0].argmax(0).item())
            images.append(X[0])
            labels.append(y.item())


    # Plot the figure 
    canvas = plt.figure(figsize=(9, 9))
    rows, cols = 5, 3
    counter = 0

    for i in range(len(labels)):

        img = images[i]
        canvas.add_subplot(rows, cols, i+1)
        plt.title("Prediction: {} Actual: {}".format(preds[i], labels[i]))
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
        plt.tight_layout()
    plt.show()

# Main function 
def main(argv):
    # Seeds used for repeatability
    random_seed = 42
    torch.backends.cudnn.enabled = False # type: ignore
    torch.manual_seed(random_seed)

    # Main training variables
    epochs = 25
    learning_rate = 1e-2
    momentum = 0.5
    
    # Load the model
    transferModel = torch.load("models/model.pth")

    # Freezes the parameters for the whole network
    for param in transferModel.parameters():
        param.requires_grad = False

    # Change the final layer to 5 outputs
    transferModel.cvstack[10] = nn.Linear(50, 5)
    # Print the model
    # print(transferModel)
    # print()
    # Load the greek train data
    greekTrainData = loadData("GreekLetters/extendedGreekTrain", 5)
    greekTestData = loadData("GreekLetters/extendedGreekTest", 1)
    # Set the loss function and optimizer for training
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(transferModel.parameters(), lr=learning_rate, momentum=momentum)
    # Run the train network on greek data
    train_network_transfer(transferModel, greekTrainData, epochs, optimizer=optimizer, lossFunction=loss_fn)
    testNewGreekLetters(transferModel, greekTestData)

if __name__ == "__main__":
    main(sys.argv)