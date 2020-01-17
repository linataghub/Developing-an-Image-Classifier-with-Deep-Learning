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

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    criterion = checkpoint['criterion']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']
    epochs = checkpoint['epochs']
        
    return model, checkpoint['class_to_idx']

def process_image(image):

    image = Image.open(image)
    
    # resize image keeping the aspect ratio
    image.thumbnail([256, 256], Image.ANTIALIAS)
    
    # crop out the center 224x224 portion of the image
    left = (image.width - 224)/2
    top = (image.height - 224)/2
    right = (image.width + 224)/2
    bottom = (image.height + 224)/2

    pil_image = image.crop((left, top, right, bottom))
    
    # color channels of images as integers 0-255,
    np_image = np.array(pil_image)/255
    
    # normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (np_image - mean)/std
    
    # reorder dimensions
    image_normalized = image_normalized.transpose((2, 0, 1))
    
    return image_normalized

# select random image
def random_image(path, cat_to_name):
    cat = np.random.choice(list(cat_to_name.keys()))
    cat_folder = path + '/' + cat + '/'
    image = cat_folder + np.random.choice(os.listdir(cat_folder))
    return image

def predict(image_path, model,  cat_to_name, topk):
    
    # process image 
    image = process_image(image_path)
    
    #Convert a JpegImageFile to a tensor 
    image = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(0)
    
    with torch.no_grad():
        
        #calculate class probabilities
        output = model.forward(image)
        prob = torch.exp(output)
        
        # Find the top K largest values
        probs, indices = torch.topk(prob, topk)
        
        # convert tensor to np array
        probs = probs.detach().numpy().tolist()[0]
        indices = indices.detach().numpy().tolist()[0]
        
        # convert the indices to the actual class labels
        idx_class_map = {val: key for key, val in model.class_to_idx.items()}
        classes = [idx_class_map[index] for index in indices]
        flower_name = [cat_to_name[idx_class_map[index]] for index in indices]
        
    return probs, classes, flower_name
    

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('data_directory', help="Directory containing the dataset")
    parser.add_argument('--checkpoint', default='checkpoint.pth', help='Checkpoint to load')
    parser.add_argument('--top_k', type=int, default=3, help='Top_k')
    parser.add_argument('--category_names', default='cat_to_name.json', help='Category names')
    parser.add_argument('--gpu', default="gpu", help="gpu")
    args = parser.parse_args()
    
    # Load model checkpoint
    model_loaded, class_to_idx = load_checkpoint(args.checkpoint)
    
    # Label mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Check for the GPU availability
    device = torch.device("cuda:0" if args.gpu =="gpu" else "cpu")
    model_loaded.to(device)
    
    # Select random image
    test_dir = args.data_directory + '/test'
    image = random_image(test_dir, cat_to_name)
    
    # Predict
    model_loaded.to('cpu')
    print(predict(image, model_loaded, cat_to_name, topk=args.top_k))
    
if __name__ == "__main__":
    main()

    