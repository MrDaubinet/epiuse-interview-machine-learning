# EPI-USE Advance Transfer learning - AI Code

### Description
This repository includes the AI code used to solve the Transfer learning interview assignment for EPI-USE Advance. 

### Integration Details
The result of this code is a classification model which is intended to deployed by a firebase serverless function.

## Technology used
* Keras
* Jupyter Notebook 
* Docker
* pandas
* matplotlib

# Implementation Details
## Part 0 - Environment Setup
I currently have 2 GTX 1070's on my personal computer which I would like to take advantage of for increased model training time. For ease of development, I would like to develop inside a docker container that has all my environment requirements setup. To use my graphics cards inside a docker container, I need to run the nvidia/cuda:10.2-base image, note that the GTX 1070 requirs cude 10.2.

The environment setup is I used is as follows: I am running windows 11, with WSL Ubuntu installed and I am running docker backed by WSL with the gpu flag enabled. I am using the [tensorflow:nightly-gpu-jupyter](https://hub.docker.com/layers/tensorflow/tensorflow/nightly-gpu-jupyter/images/sha256-1a28ddb075916cdc1e5660b4b27782e08a0da34cf8dc3652fc9fbf1519dc9f5b?context=explore) base image.

I used vscode to open my folder within a docker container. Once my images have been built and the IDE is interactable, I opened a new terminal and ran ```jupyter notebook --allow-root```.

## Part 1 - Model
After attempting to use transfer learning with a few [pretrained keras models](https://keras.io/api/applications/), I came to fast success with my training accuracy by using the [Xception](https://arxiv.org/abs/1610.02357) model. The other perk about this model is that it was small enough to fine tune (with all parameters trainable) on the available GPU memory of my system (more on this in part 6). 

My tranfer learning model had the following structure:
![alt text](https://github.com/MrDaubinet/epiuse-interview-machine-learning/blob/master/Images/model_summary.jpg?raw=true)

## Part 2 - Dataset
I chose the **Dog Breed Identification** dataset which can be manually downloaded from Kaggle [here](https://www.kaggle.com/c/dog-breed-identification/data). I chose this dataset for multiple reasons. Its large, with over 10 000 training samples. Its thouroughly tested for this task, being linked to a competition which has over 1200 entries from different users. It's cool, you can go take images of you own dogs and see if the trained model works. 

The Dog Breed Dataset has 10 222 training samples, 10 357 test sample for 120 categories. However, due to the nature of this dataset, the test samples are not labled and therefore will not be used to measure model performance. Instead, I chose to create a keras ImageDataGenerator from training dataset, with this data split into training and validation at 80:20 ratio. 

![alt text](https://github.com/MrDaubinet/epiuse-interview-machine-learning/blob/master/Images/training_data.jpg?raw=true)

## Part 3 - Data prep

The only data prep needed was to great a ImageDataGenerator from a dataframe and directory 
```
# create the tensorflow image generator for train and validation sets
train_data_gen = ImageDataGenerator(
        rescale = 1./255, 
        fill_mode='nearest', 
        validation_split = 0.2)

# Training Generator
train_generator = train_data_gen.flow_from_dataframe(
    df_train, 
    directory = train_dir, 
    x_col = 'id', 
    y_col = 'label', 
    subset = 'training',
    class_mode = 'categorical', 
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE, 
    shuffle = True, 
    seed = 123)

# Validation Generator
validation_generator = train_data_gen.flow_from_dataframe(
    dataframe=df_train, 
    directory = train_dir, 
    x_col = 'id', 
    y_col = 'label',
    subset = 'validation', 
    class_mode = 'categorical', 
    target_size = (IMG_SIZE, IMG_SIZE), 
    batch_size = BATCH_SIZE, 
    shuffle = True, 
    seed = 123)
```

## Part 4 - Data Augmentation
I implemented as simple augmentation layer that was then added as the first input to my transfer leraning model. The augmentation details are as follows:

```
img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)
```

![alt text](https://github.com/MrDaubinet/epiuse-interview-machine-learning/blob/master/Images/augmentation_data.jpg?raw=true)

## Part 5 - Training & Test Results
The tranfer learning model learnt to classify my data, fairly accuratly, from early on. 

![alt text](https://github.com/MrDaubinet/epiuse-interview-machine-learning/blob/master/Images/training_epochs.jpg?raw=true)

I set my training epochs to a max of 30, however my model stoped training after 15 epochs due to an early stopping restriction which was imposed following 10 epochs with no gains in the validation loss. My model ended training with a training accuracy of 96% and a validation accuracy of 89%. For the purposes of this assignment, I treated validation accuracy the same as testing accuracy. I made no model selection choices based on the validation accuracy and I gathered no further testing results.

![alt text](https://github.com/MrDaubinet/epiuse-interview-machine-learning/blob/master/Images/model_accuracy.jpg?raw=true)

## Part 6 - Fine Tuning
In an attempt to make the model more accurate against my validation/test set. I unfroze all model layers and applied the SGD optimizer with a low learning rate for a further 10 epochs. Unfortunately, this only resulted in the model becoming more over fit, with an increase in the training accuracy and a decrease in the validation / test accuracy.

![alt text](https://github.com/MrDaubinet/epiuse-interview-machine-learning/blob/master/Images/fine_tuning.jpg?raw=true)

## Jupyter Notebook
