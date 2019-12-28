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
from tabulate import tabulate
from torch.autograd import Variable

def main():
    # Variable initiation
    checkpoint = 'checkpoint.pth'
    filepath = 'cat_to_name.json'    
    image_path = 'flowers/test/7/image_07211.jpg'
    topk = 5

    # Command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store',type=str, help='Name of trained model to be loaded and used for                 predictions.')
    parser.add_argument('-i','--image_path',action='store', default='flowers/test/7/image_07211.jpg',type=str, help='Image directory to predict from')
    parser.add_argument('-k', '--topk', action='store',type=int, help='Choose the top classes')
    parser.add_argument('-j', '--json', action='store',type=str, help='Define name of json file holding class names.')
    parser.add_argument('-g','--gpu', action='store_true', help='Use GPU if available')


    args = parser.parse_args()

    # Command line parameters
    if args.topk:
        topk = args.topk
    if args.checkpoint:
        checkpoint = args.checkpoint
    if args.image_path:
        image_path = args.image_path
    if args.json:
        filepath = args.json
    if args.gpu:
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)



    # loading saved checkpoint
    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model = checkpoint['model']
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.epochs = checkpoint['epochs']
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer = checkpoint['optimizer']

        return model
    file_name = 'checkpoint.pth'
    model = load_checkpoint(file_name)
    print(model)
    
    # processing image function
    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''

        #Process a PIL image for use in a PyTorch model(PIL was imported as Image)
        pic = Image.open(image)
        pic = pic.resize((256,256))
        value = 0.5*(256-224)
        pic = pic.crop((value,value,256-value,256-value))
        pic = np.array(pic)/255

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        pic = (pic - mean) / std

        return pic.transpose(2,0,1)

    # prediction function
    def predict(image_path, model, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''

        # TODO: Implement the code to predict the class from an image file
        # move the model to cuda
        cuda = torch.cuda.is_available()
        model.cuda()


        # turn off dropout
        model.eval()

        # The image
        image = process_image(image_path)

        # tranfer to tensor
        image = torch.from_numpy(np.array([image])).float()

        # The image becomes the input
        image = Variable(image)
        if cuda:
            image = image.cuda()

        output = model.forward(image)

        probabilities = torch.exp(output).data

        # deriving the topk (=5) probabilites 
        prob = torch.topk(probabilities, topk)[0].tolist()[0] 
        index = torch.topk(probabilities, topk)[1].tolist()[0] 

        pp = []
        for i in range(len(model.class_to_idx.items())):
            pp.append(list(model.class_to_idx.items())[i][0])

        # converting id to label
        label = []
        for i in range(5):
            label.append(pp[index[i]])

        return prob, label


    model = load_checkpoint(checkpoint) 

    print(' ')
    print(' ')
    print('The table below shows the flower name and their respective probabilities.')
    print(' ')
    prob, classes = predict(image_path, model,topk)
    # print(prob)
    # print(classes)
    # print([cat_to_name[x] for x in classes])
    print(tabulate([((prob))], headers=[cat_to_name[x] for x in classes]))
    
if __name__== "__main__":
  main()