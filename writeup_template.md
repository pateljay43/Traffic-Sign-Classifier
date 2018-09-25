# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/random_data_set.png "Visualization"
[image2]: ./examples/bar_chart.jpg "Bar Chart"
[image3]: ./examples/pre_processed.jpg "Pre Processed Random Images"
[image4]: ./examples/00001.png "Traffic Sign 1"
[image5]: ./examples/00093.png "Traffic Sign 2"
[image6]: ./examples/00107.png "Traffic Sign 3"
[image7]: ./examples/00130.png "Traffic Sign 4"
[image8]: ./examples/00166.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

##### Code Cell #: 2

I used the numpy library to calculate statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

##### Code Cell #: 3

Here is an exploratory visualization of the data set. It is a random set of signs from dataset ...

![alt text][image1]

##### Code Cell #: 4

This bar chart represents count of each sign type ...

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

##### Code Cell #: 5

As a first step, I decided to normalize images because by doing so we will get zero mean and equal variance which proves helpful in Stochastic Gradient Descent process. The catch here is the input type where by its uint8 by default, and that does not work well because negative results get wraped around to keep it within range of 0...255. Hence we have to convert it to a float type ...

After that, I converted those normalized images to grayscale by using the average method described at: "https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/"

Here is an example of a random traffic signs after pre-processing them.

![alt text][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

##### Code Cells: 7, 8, 9, 10, 11

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 1x1	    | 2x2 stride, valid padding, outputs 1x1x412    |
| RELU					|												|
| Fully connected		| input 412, output 122        									|
| RELU					|												|
| Dropout				| 50% keep prob        									|
| Fully connected		| input 122, output 84        									|
| RELU					|												|
| Dropout				| 50% keep prob        									|
| Fully connected		| input 84, output 43        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

##### Code Cells: 12, 13

To train the model, I used an Adam optimizer, a batch size of 150, 20 epochs, and a learning rate of 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

##### Code Cell: 15

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.949
* test set accuracy of 0.932

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
- Initially I started with similar architecture offered in LeNet Lab Solution workspace. But that was not enough. Hence did some research and found adding more covnet network would bump up the accuracy and it did.
* What were some problems with the initial architecture?
- The original model was not able to learn well maybe because of the depth of layers.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
- I had to add another covnet layer to get better accuracy for all validation dataset. Training dataset was working find without this modification. Hence I guess original model was over fitting.
* Which parameters were tuned? How were they adjusted and why?
- Had to tune up a bit all the given parameters epochs, learning rate, batch size
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
- I did reuse most of the part from LeNet Lab Solution but from my understanding it was over fitting the model and it required more covnet layers to address that issue. Dropout layer could be helpful if we had much more and a well distributed dataset.

If a well known architecture was chosen:
* What architecture was chosen?
- LeNet model with another covnet layer.
* Why did you believe it would be relevant to the traffic sign application?
- This model learns parts or I should say patches from the images and uses those patches to identify new images. This helps to match a sign in one image which could be located anywhere (cornor, or center) with another one of same type but having that sign at a totally different place. This is possible due to covnet.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
- Final model's accuracy are very close when compared to different datasets. They all are within 7 percent range which is okay according to me. But it still has scope to improve with more tuneup to those parameters and the architecture itself.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

##### Code Cells: 16, 17, 18

Here are five German traffic signs that I found on the web. They all are very common sign images and my model was able to classify all of them with 100% accuracy for each of them:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h      		| 30 km/h   									| 
| Stop Sign     			| Stop Sign 										|
| Road work					| Road work											|
| Go straight or right	      		| Go straight or right					 				|
| Yield			| Yield      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares way more favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

##### Code Cell: 19

As mentioned above, the 5 signs from German Traffic Signal dataset were very clear and hence the model was able to classify them very easily. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| 30 km/h   									| 
| 1.0     				| Stop Sign 										|
| 1.0					| Road Work											|
| 1.0	      			| Go straight or right					 				|
| 1.0				    | Yield      							|


