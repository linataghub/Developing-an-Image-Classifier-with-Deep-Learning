import argparse
import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import time
import copy
from collections import OrderedDict
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from PIL import Image
import json

def data_prep(data_dir):
    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    

    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'valid', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

    return dataloaders, dataset_sizes, image_datasets

def build_model(arch, learning_rate, hidden_units):
    
    if arch == 'vgg': 
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features

    for param in model.parameters():
        param.requires_grad = False
        

    # Number of classes represented by the dataset
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    num_classes = len(cat_to_name)

    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()), 
        ('dropout', nn.Dropout(p=0.4)),
        ('fc2', nn.Linear(hidden_units, num_classes)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    # Replace the model classifier with this new classifier 
    model.classifier = classifier
      
    #Define criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer =  optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    return model, criterion, optimizer, scheduler

def train_model(model, criterion, optimizer, scheduler, epochs, device, dataloaders, dataset_sizes):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                    
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    parser = argparse.ArgumentParser(description='Train network on a dataset')
    parser.add_argument('data_directory', help="Directory containing the dataset")
    parser.add_argument('--save_dir', default="checkpoint.pth", help="Save checkpoint directory")
    parser.add_argument('--arch', default="vgg", choices={"vgg", "densenet"}, help="Pre-trained network: vgg or densenet")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--hidden_units', type=int, default=512, help="Hidden unit")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    parser.add_argument('--gpu', default="gpu", help="gpu")
    args = parser.parse_args()
    
    # Prepare data
    dataloaders, dataset_sizes, image_datasets = data_prep(args.data_directory)
    
    # Load pre-trained network
    model, criterion, optimizer, scheduler = build_model(args.arch, args.learning_rate, args.hidden_units)
    
    # Check for the GPU availability
    device = torch.device("cuda:0" if args.gpu =="gpu" else "cpu")
    model.to(device)
    
    # Train network
    model = train_model(model, criterion, optimizer, scheduler, args.epochs, device, dataloaders, dataset_sizes)
    
    # Save checkpoint
    
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {'epochs': args.epochs,
                'model': model,
                'criterion': criterion,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler,
                'model_state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx
                 }

    torch.save(checkpoint, 'checkpoint.pth')

if __name__ == "__main__":
    main()


