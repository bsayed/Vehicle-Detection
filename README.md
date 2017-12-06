
# Vehicle Detection Project

The steps/goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier.
* Added a color histogram, to the HOG feature vector. 
* Normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream to create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for detected vehicles .

[//]: # (Image References)
[image1]: ./examples/bbox_1.jpg
[image2]: ./examples/bbox_2.jpg
[image3]: ./examples/bbox_3.jpg
[HOG]: ./examples/HOG_example.jpg
[vehicle]: ./examples/vehicle.jpeg
[not_vehicle]: ./examples/not_vehicle.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is implemented by the `get_hog_features()` method, in lines 42 through 59 of the file called `HelperFunctions.py`.  
This is a static method of the `VehicleDetection` class. 
I started by reading in all the `vehicle` and `non-vehicle` images.  
Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][vehicle]

![alt text][not_vehicle]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][HOG]

#### 2. Explain how you settled on your final choice of HOG parameters.

In order to settle on the final choice of HOG parameters, I trained a linear SVM classifier on the small vehicle-non-vehicle
dataset provided by Udacity using different combinations of parameters settling on the combination that scored
the highest test accuracy. 


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained the linear SVM using example images that come from a combination of 
the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), 
the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), 
and examples extracted from the project video itself.
The training is implemented in lines 94 through 138 in `SearchClassify.py` file, the code extracts the features
from both the vehicles and non-vehicles examples then normalize the feature vectors, then randomly splits the
feature vectors into training and testing data, with 70% and 30% ratio respectively. Finally, after training
the linear SVM classifier, the classifier and the scalar object created as the result of the normalization step are stored in a pickled file.
This is for the purpose of not training the classifier every time we need to run the vehicle detection algorithm.
The scalar object is used later to normalize the features extracted from test images in the same way it normalized
the feature in the training step. Otherwise, the classifier performance and accuracy will be greatly affected.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
After several iterations on the sliding window sizes verses the overlap, I have found that smaller window size
tend to produce higher false positives than bigger sliding windows. In addition, they take longer time to scan
the image. I found that the 64x64, 96x96, 128x128 window sizes to be the right balance between performance, 
accuracy, and low false positive.
I used the HOG sub-sampling technique to speed up the prediction process, my initial implementation used to 
take more than 10 seconds per frame, which is not a very good performance, the HOG sub-sampling takes 
on average 1.25 seconds, which is clearly a better choice, in addition to that, I have implemented my code in
a way that every window size (scale) is run in a separate thread, so instead of running the scanning process
sequentially, the code spawns a thread for each window size (scale), this is outlined in lines 303 through 320
in file `HelperFunctions.py`. The goal was to achieve several frames per second, but apparently the thread.join()
method affects the performance a little and the max speed I was able to achieve is one frame per seconds.
It is important to note here that the code only scans the lower half of the image, this is where we expect to
see vehicles.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales, 64x64, 96x96, 128x128 using YCrCb 3-channel HOG features plus 
histograms of color in the feature vector, which provided a good performance. 
I ignored the spatially binned color information as it did not improve the accuracy by much.
One technique that I used to optimize the performance of the sliding window search was to bound 
the search area of the subsequent frames based on what we found in the first frame. The implementation is
outlined in the `HelperFunctions.py` file, in the function `draw_labeled_bboxes()` where I keep the latest
starting and ending positions in both the x and y axes based on what we found in this particular frame.
Then I enlarge the this bounding box by 10% (line 253 and 260, `HelperFunctions.py`) to use it as the new
search area for the next frame, the intuition is that the cars we found in this frame will not move by much
in the next frame. However, we have to reset the search area every once in a while to pick up vehicles that 
were not detected before or just entered the scene. I reset the search area to be the entire lower half
of the image every 25 frames (equivalent to one second in the project video), this is demonstrated in 
line 296 and 297 in `HelperFunctions.py`, this is also were we reset the heat map, more on the heat map
down below.

Here are some example images:

![alt text][image1]

![alt text][image2]

![alt text][image3]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I gathered the positions of positive detections in each frame of the video from the different threads, 
where each thread represent a different scale (window size), line 322 through 326 file `HelperFunctions.py` .
From the positive detections I created a heatmap and then thresholded that 
map to identify vehicle positions and eliminate the false positive, I used a threshold value of 3, which means
three overlapping windows in any given position to consider it a correct detection, otherwise, it is a 
false positive.
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap, line 340 in 
`HelperFunctions.py` file.
I then assumed each blob corresponded to a vehicle.
I constructed bounding boxes to cover the area of each blob detected, line 341 in `HelperFunctions.py` file.  

Here's an example result showing the heatmap from a series of frames of video, 
the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on 
the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue in the current implementation in my opinion, is the performance. I think 2 or 3 frames 
per second is not enough for an actual working self-driving car. I tried to tackle this challenge
by using a thread for each window size, however, did not perform that well, ideally 15 or more frames per
second would be something reasonable, let a lone the fact that other processing is already happening at the
same time, like lane detection or pedestrian detection, etc. One possible way of improving the performance is
to utilize GPUs to do some or most of the work.


