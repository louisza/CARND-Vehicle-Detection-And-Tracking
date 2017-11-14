

**Vehicle Detection Project**  
***Louis Pienaar***


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[Car_NonCar]: ./output_images/Car_NonCar.png
[CarHog]: ./output_images/CarHog.png
[SlidingWindow]: ./output_images/SlidingWindow.png
[InitialPipeline]: ./output_images/InitialPipeline.png
[Heatmap]: ./output_images/Heatmap.png
[Threshold]: ./output_images/Threshold.png
[Label]: ./output_images/Label.png
[FinalPipeline]: ./output_images/FinalPipeline.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell 3 of the IPython notebook.  

As a start, I read in the car and non-car images provided for the project
![alt text][Car_NonCar]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). 
I iteratively went through a set of parameter combinations combined with the SVC model results. The parameters quickly converged to a point where teh SVC models were performing extremley well.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:


![alt text][CarHog]

####2. Explain how you settled on your final choice of HOG parameters.

After starting off with the default parameters in the lesson, I moved the aparamters iteratively into directions to see the effect of the parameter, once I had a good feel for what I think would be provide a good set of featuires, I tested the parameters using accuracy of my SVC model as a benchmark.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

My linear SVC (linear support vector classifier) model was trained on a random train/test split of the car/non-car images. The HOG features of each image were extracted and the scaled.
The code for these steps can be found in cells 6 to 9.

My final SVC model, after the parameter tuning, had an accuracy of 0.989 on the test set.



###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

By inpsecting the video you quickly realise that your sliding windows that you should use to search for cars can be confined to a portion of the image only. (There should be no cars in the sky, if tehre are, you have bigger problems to deal with).
Also, the size(scale) of the window differs, as the lower you go in teh image, the closer a car would be, therefore the bigger your window should be.

Here is an image showing the results of the sliding window grid that I ended up using to search for cars.

Essentially each one of these windows would be put through the SVC to predict if a car is in the window or not.

![alt text][SlidingWindow]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The initial results of the SVC and my search grid was performing OK. I played around with the HOG parameters and also the start stop position of the search grid and got to a point where I could see that the results are accurately finding the cars on my test image.
Here is the result of the initial pipeline on a test image. Notice teh false positive on the left hand side of the image.

![alt text][InitialPipeline]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)

The pipeline does perform reasonable well, there are a number of concerns that I will address at the end of this write up

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and
 some method for combining overlapping bounding boxes.

I implemented a VehicleDetector class to keep record of the last few images. I also used a heatmap method
 together with a threshold to get rid of false positives. The reason why this works is because from inspection you
  will notice that true positives almost always have more than one overlapping window classified as car, 
  whereas false positives often only have one window.
A heatmap helps define regions with overlapping windows. By applying a threshold you effectively eliminate 
false positives.

Here is the heatmap for my test image

![alt text][Heatmap]

After applying a threshold of 1, (There should be at least two overlapping windows). 
Notice that the false positive have been eliminated.

![alt text][Threshold]

The using the label function, we can group (or label) adjacent values, essentially combing overlapping windows into one.

![alt text][Label]

####3. Using all of the above, the final pipeline applied onto the test images:

![alt text][FinalPipeline]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This pipeline is more of a proof of concept and will most likely fail when applied on situations that are not so 
controlled. For example, adverse lighting and weather. The solution to this would be to get more data and features to 
be used in the SVC model.

The boxes also does not accurately bound the vehicles and do not track multiple vehicles correctly when the vehicle goes 
beyond another. A possible solution for this would be to use the current pipeline to identify the area where a car is 
and then use another pipeline to focus on that area and accurately define the car boundaries. Using a Hog outline 
in that area might be a neat way to snap onto the vehicle outlines.

The pipeline is also fairly slow at the moment, part of why I chose to use a class for my pipeline is to enable 
better search grids.
For example, I could confound the search area to where a vehicle was found previously, 
and only do a full search every x amount of frames.

The speed of the pipeline would be crucial in a real world implementation.
 Luckily I know that, that is a whole field of study on it's own.
Advances like the recent Nvidia TensorRT 2 development that optimises trained neural networks for deployment is proof 
of how important this is.


 
 