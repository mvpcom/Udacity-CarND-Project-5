**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./element/datasetVisualization.png
[image11]: ./element/datasetVisualization2.png
[image2]: ./element/hogChannel2.png
[image21]: ./element/hogChannel2-1.png
[image3]: ./element/hogChannel1.png
[image31]: ./element/hogChannel1-1.png
[image4]: ./element/hogChannel0.png
[image41]: ./element/hogChannel0-1.png
[image5]: ./element/colorHistogram.png
[image51]: ./element/colorHistogram-1.png
[image6]: ./element/simpleSliding.png
[image7]: ./element/3.png
[image8]: ./element/2.png
[image9]: ./element/1.png
[image10]: ./element/index.png
[image15]: ./element/6frameHeatmap.png
[image16]: ./element/6frame.png
[video1]: ./output_videos/p5VideoMine.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the `get_hog_features` function. first code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of some of each of the `vehicle` and `non-vehicle` classes:

![visualization][image1]
![visualization][image11]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are some examples using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Channel 3:
![alt text][image2]
![alt text][image21]

Channel 2:
![alt text][image3]
![alt text][image31]

Channel 1:
![alt text][image4]
![alt text][image41]


![alt text][image5]
![alt text][image51]


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally I find this parameters are my best choice: 
```
color_space = 'YCrCb' #HSV:99 # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM combined with a feature selection and sigmoid for probability output. You can find it under "Best Model is here: SVM with Feature selector and probability output" Header in the jupyter notebook.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to implement a more intelligent way to find sliding windows. Although I implemented that successfully but I found out the result of my manual scaling and windows search with this parameters:

```    
windows1 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                xy_window=(64, 64), xy_overlap=(0.5, 0.5))
windows2 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                xy_window=(95, 95), xy_overlap=(0.5, 0.5))
windows3 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                xy_window=(58, 58), xy_overlap=(0.5, 0.5))
windows = windows1 + windows2 + windows3
```

![alt text][image6]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
---


### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/p5VideoMine.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

To filter false positives, I used techniques such as probability determination of classifier, heatmap and threshold value and average heatmap.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image15]


### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image16]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think the most valuable things that I need was time to make it better. I need to pass term 1 and I'm currently developing other ideas such as deep learning. But the most interesting part I wanna work there is make the algorithm realtime and find some technique to replace sliding windows search or at least make it more intelligently. The speed is one of the most important thing here and I need to make sure my algorithm perform fast and well enough in the real world. I think my pipeline can fail easily with more shadowy and more comlicated scene. It will not work in night or all weather situations. There is more work to do to make this algorithm better. 
