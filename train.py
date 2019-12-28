# Imports here

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, random
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models
from torchvision.transforms import *
from collections import OrderedDict
from PIL import Image
import time
import argparse


def main():
    
    # Set up parameters for entry in command line
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',type=str, default='flowers', help='Location of directory with data for image classifier to         train and test')
    parser.add_argument('-a','--arch',action='store',type=str, help='Define pretrained network densenet121')
    parser.add_argument('-l','--learning_rate',action='store',type=float, default= 0.001, help='Choose a float number as the learning       rate for the model')
    parser.add_argument('-e','--epochs',action='store',type=int, default= 5, help='Choose the number of epochs you want to perform          gradient descent')
    parser.add_argument('-g','--gpu',action='store_true',default='cuda',help='Use GPU if available')

    args = parser.parse_args()

    # Initiate variables with default values
    arch = 'densenet121'
    learning_rate = args.learning_rate
    epochs = args.epochs
    device = args.gpu
    if args.gpu == 'cpu':
        device = "cpu"
    else:
        device = "cuda"

    # directory
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'    

    # loaders and transforms
    means = [0.485, 0.456, 0.406]
    deviations = [0.229, 0.224, 0.225]
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means,
                                                                deviations)])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means,
                                                               deviations)])

    test_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means,
                                                               deviations)])

    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transform)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transform)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    # label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    if args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    elif args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    else:
        print('sorry the model you inputed is not found, please choose between')

    # load a pretrained model
#     model = models.densenet121(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # defining classifier
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, 500)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(0.5)),
                              ('fc2', nn.Linear(500, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # select gpu or cpu
#     device = "cuda"
    model = model.to(device)
    # Implement a function for the validation pass
    def validation(model, validloader, criterion):
        test_loss = 0
        accuracy = 0
        for inputs, labels in validloader:
            inputs,labels = inputs.to(device),labels.to(device)

            output = model.forward(inputs)
            test_loss = criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return test_loss, accuracy

    # train model and track loss
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 40
    train_losses,valid_losses = [],[]
    print('Training -------')
    for e in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            inputs,labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion)

                train_losses.append(running_loss/len(trainloader))
                valid_losses.append(test_loss/len(validloader))
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                    "Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0
                # Make sure training is back on
                model.train()  

    # TODO: Do validation on the test set
#     device = "cuda"
    model = model.to(device)
    def test(model, testloader):
        model.to(device)
        model.eval()
        accuracy = 0
        for inputs, labels in testloader:
            inputs,labels = inputs.to(device),labels.to(device)

            output = model.forward(inputs)

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        print("Testing Accuracy: {:.3f}".format(accuracy/len(testloader)))

    test(model, testloader)

    #Save the checkpoint 
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': input_size,
                 'output_size': 102,
                 'batch_size': 64,
                 'model': model,
                 'epochs': epochs,
                 'classifier': classifier,
                 'optimizer': optimizer.state_dict(),
                 'state_dict': model.state_dict(),
                 'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')

if __name__== "__main__":
  main()
