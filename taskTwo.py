# Karan Shah
# Analyze first convolution layer, plot the weights, along with the filters and their effects

# Import libraries
import sys
import torch
from taskOne import MyNetwork
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
import cv2 as cv

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
    
    return trainingData



# Plot the 10 filters
def plotFilters(layer):
    canvas=plt.figure(figsize=(8, 8))  
    rows, cols = 3, 4
    with torch.no_grad():
        for i in range(10):
            filters = layer[i]
            canvas.add_subplot(rows, cols, i+1)
            plt.title(f"Filter {i}")
            plt.axis("off")
            plt.imshow(filters.squeeze())
            plt.tight_layout()
        plt.show()

# Plots effects of layer on data
def plotEffects(layer, data):
    canvas = plt.figure(figsize=(8, 8))  
    rows, cols = 5, 4
    idx = 0
    with torch.no_grad():
        for idx in range(layer.shape[0]):
            # filters is 1, 5, 5 
            filters = layer[idx]
            # Apply rhe filter to the image
            filteredImg = cv.filter2D(data[0].numpy(), -1, filters[0].numpy())
            canvas.add_subplot(rows, cols, 2 * idx + 1)
            plt.title("Filter:{}".format(idx))
            plt.axis("off")
            plt.imshow(layer[idx].squeeze(), cmap="gray")
            plt.tight_layout()
            canvas.add_subplot(rows, cols, 2 * idx + 2)
            plt.imshow(filteredImg, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            idx += 1
    plt.show()



# Main function 
def main(argv):
    # Seeds used for repeatability
    random_seed = 42
    torch.backends.cudnn.enabled = False # type: ignore
    torch.manual_seed(random_seed)

    # Load the models
    loadedNetwork = torch.load("models/model.pth")
    # print(loadedNetwork)

    # Obtain the first convolution layer
    cvstack = loadedNetwork.cvstack
    firstConvLayerWeights = cvstack[0].weight

    # # Print the shapes and weights of the first convolution layer    
    for i in range(firstConvLayerWeights.shape[0]):
        print("Shapes: {}".format(firstConvLayerWeights[i,0].shape))
        print("--------")
        print("Weights: {}".format(firstConvLayerWeights[i,0]))
        print()

    # Plot filters of first convolution layer 
    plotFilters(firstConvLayerWeights)

    # Load the data
    trainingData = loadData()

    # First 0 is to get first example. Second 0 to get X. Result is 1 x 28 x 28.
    firstImage = trainingData[0][0]

    # Plot the effect of the first convolution layer on the first training image
    plotEffects(firstConvLayerWeights, firstImage)

if __name__ == "__main__":
    main(sys.argv)