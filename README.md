# Recognition Using Deep Networks 

This project is about learning how to build, train, analyze, and modify a deep network for a recognition task. We will be using the MNIST digit recognition data set, primarily because it is simple enough to build and train a network without a GPU, but also because it is challenging enough to provide a good example of what deep networks can do.

## Author: Karan Shah

If any issues, contact me at shah.karan3@northeastern.edu 

## Environment

OS: MacOS M1 Pro  
IDE: Visual Studio Code  
Python version: 3.9.12  

# How to run the code

## To train the model and save the networks (Tasks 1A-1E)
python taskOne.py

## To load and run on new inputs (Tasks 1F-1G)
python taskOneRemainder.py

## To examine the network (Task 2)
python taskTwo.py

## To perform transfer learning on greek letters (Task 3)
python taskThree.py

## To design the experiment and hyperparameter tune the model (Task 4)
python taskFour.py

## To run the additional greek letter extensions (extension.py is just adaption of Task 3 code)
python extension.py

## Where to execute
Run all these executables one level down in the root directory. Keep the data folders and executables all here. 

## Additional Folders
figures: Contains the results  
GreekLetters: Contains alpha, beta, gamma, omega, and delta greek letters for different tasks (needed for taskThree.py and extension.py)     
models: Saves the model  
newestDigits: Contains personal handwritten digits (needed for taskOneRemainder.py)    

