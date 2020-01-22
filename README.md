# Developing-an-Image-Classifier-with-Deep-Learning

In this project, we will train an image classifier to recognize different species of flowers. We will be using this dataset of [102 flower categories](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

The project is broken down into multiple steps:

- Load and preprocess the image dataset
- Train the image classifier on the dataset
- Use the trained classifier to predict image content

## Installment
The project includes two files train.py and predict.py. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image.

Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu

### Acknowlegdment

**Project context:** Udacity Data Scientist Nanodegree program
**Dataset:** [102 flower categories](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).
