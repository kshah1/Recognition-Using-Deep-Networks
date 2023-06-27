# Karan Shah
# Load the model and evaluate on new written digits

# Import library
import sys
from taskOne import MyNetwork
from torchvision import datasets, transforms
import torch
from matplotlib import pyplot as plt
import cv2 as cv 
import numpy as np

# Loads the test data
def loadData():
    testData  = datasets.MNIST(
                            root="data",
                            train=False,
                            download=True,
                            transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                      (0.1307,), (0.3081,))
                            ])
                    )
    return testData

# Plots the first ten examples in the test set
def plotEvaluation(data, predictions, labels):
    # data is tensor: 10, 1, 28, 28
    # prediction is tensor: 10, 10
    # label is tensor: 10, 1

    # Create 12 by 12 figure
    canvas = plt.figure(figsize=(8, 8))

    cols, rows = 3, 3
    # Select 9 gray scale test images and add to canvas
    for i in range(cols * rows):
        canvas.add_subplot(rows, cols, i+1)

        img, prediction, label = data[i], predictions[i], labels[i]

        plt.title("Prediction: {} Actual: {}".format(prediction, label))
        canvas.add_subplot(rows, cols, i+1)
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
    # Load the test data
    testData = loadData()
    # Load the model
    loadedModel = torch.load("models/model.pth")
    # Set to evaluation
    loadedModel.eval()


    # Uncomment up to line 90 to plot the first ten examples in the test set after loading the model in 

    # images = []
    # predictions = []
    # labels = []

    # # Wrapping dataloader around test set to obtain batch of 1
    # # X is tensor: 1, 1, 28, 28
    # idx = 0
    # with torch.no_grad():
    #     for X, y in torch.utils.data.DataLoader(testData): # need to reload the test data in seperate script for this part
    #         if idx == 10:
    #             break
    #         # pred is tensor: 1, 10
    #         pred = loadedModel(X)
    #         images.append(X[0])
    #         predictions.append(pred[0].argmax(0).item())
    #         # label is tensor: 1
    #         labels.append(y[0].item())
    #         torch.set_printoptions(precision=2)
    #         print("Probabilities:", pred)
    #         print("Index of max output:", pred[0].argmax(0).item())
    #         print("Correct Label:", y[0].item())
    #         print()
    #         idx += 1
            
    # plotEvaluation(images, predictions, labels)

    newInputs = []
    newLabels = []
    newPredictions = []
    # Iterate through the new digits and preprocess
    for i in range(10):
        image = cv.imread("newestDigits/cropped/{}_cropped.png".format(i))
        grayScaleImg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        invertedImg = cv.bitwise_not(grayScaleImg) # flip pixel values
        invertedImg = invertedImg/255 # intensities
        expandedImg = np.expand_dims(invertedImg, axis = 0)
        tensorImg = torch.tensor(expandedImg, dtype=torch.float32)
        newInputs.append(tensorImg)
        newLabels.append(i)
        
    # # Evaluate model on the new digits
    for X in torch.utils.data.DataLoader(newInputs): 
        with torch.no_grad():
            pred = loadedModel(X)
            newPredictions.append(pred[0].argmax(0).item())
            


    # # After testing the first 10 examples, plot them out
    canvas=plt.figure(figsize=(8,8))  
    rows, cols = 3, 4
    for i in range(10):
        img = newInputs[i]
        canvas.add_subplot(rows, cols, i+1)
        plt.title(f"Prediction: {newPredictions[i]} Actual: {newLabels[i]}")
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
        plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sys.argv)