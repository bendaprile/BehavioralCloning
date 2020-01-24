# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* autonomous_track1.mp4 a video of the output

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 52-65) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 53-59). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 62). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 68). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 67).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the one nvidia uses for their self-driving cars. I thought this model might be appropriate because nvidia uses it for behavioral cloning on their self-driving vehicles.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that there was a dropout layer between my first and second fully connected layers. My dropout rate was 0.4 percent because this performed the best for me.

Then I finished up with a few more fully connected layers to get my final output from the network of 1.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. Specifically my vehicle liked to run into the wall on the bridge and go off the track right after that where the dirt was. To improve the driving behavior in these cases, I added the dropout layer and recorded more training data around these sections of track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 

My model starts out with a Keras lambda layer to normalize all of my data. I then added a cropping layer to crop the images to what I wanted. This removed the top portion of trees and the bottom portion containing the car hood. (model.py lines 53-54)

I then use five 2D convolutional layers with 5x5 filters and 3x3 filters. I use the RELU activation for all of these convolutional layers to introduce nonlinearity.

Lastly I flatten the model and use the Dense function to flatten it to 100. I then go through a combination of fully connected layers until the output of my neural network is 1, the steering angle.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. These pictures were saved in the mydata folder.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if it started wandering close to the lanes. These images were also saved in the mydata folder.

To augment the data sat, I also did a couple laps around the track in the opposite direction to better generalize the data and give the car more of a sense of when to turn right.

After all of this my car still ran into the wall on the bridge and went off the track near where the lines stop and the dirt starts. I recorded a few more data points near these areas to help the car understand them.

After the collection process, I had many data points. I then preprocessed this data by normalizing and cropping every image. I used both udacity's provided data and my own data for this project to have more data.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by the testing I did of many different epochs. Five was not enough as I could still lower my validation loss and 7 was too high as the validation loss started to go back up. I used an adam optimizer so that manually training the learning rate wasn't necessary.

In the near future I plan on trying to get this to work on the second track as it does not understand what to do on the second track right now.