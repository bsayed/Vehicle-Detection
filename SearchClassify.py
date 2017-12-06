import argparse

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from HelperFunctions import *
import pickle
from moviepy.editor import VideoFileClip

from sklearn.model_selection import train_test_split


# test_images = glob.glob('test_images/test*.jpg')
#
# for image_file in test_images:
#     image = mpimg.imread(image_file)
#     t_start = time.time()
#     window_img = vd.process_frame(image)
#     t_end = time.time()
#     print(round(t_end - t_start, 2), ' Seconds to process frame...')
#     window_img = cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB)
#
#     cv2.imwrite(image_file.split('.')[0] + "-Threads.jpg", window_img)



def main():
    """Parses the command line options and kick-start the pipe-line to detect vehicles in videos."""

    parser = argparse.ArgumentParser()

    parser.add_argument("-hgt", "--imgHeight", help="The height of the images, default=720.",
                        type=int, default=720)

    parser.add_argument("-wd", "--imgWidth", help="The width of the images, default=1280.",
                        type=int, default=1280)

    parser.add_argument("-in", "--inputVideoPath", help="The path to the input video to be processed.",
                        type=str, default='')

    parser.add_argument("-out", "--outputVideoPath", help="The path to the where to store output video.",
                        type=str, default='')

    parser.add_argument("-tm", "--trainingMode", help="Run the program to train the SVM classifier",
                        type=bool, default=False)

    parser.add_argument("-c", "--cars", help="The path to the cars training images. Can be in glob format.",
                        type=str, default='')

    parser.add_argument("-notc", "--notcars", help="The path to the NOT cars training images. Can be in glob format.",
                        type=str, default='')

    args = parser.parse_args()

    print(args)

    assert args.inputVideoPath != '', "The path to input video can't be empty"
    assert args.outputVideoPath != '', "The path to output video can't be empty"

    if args.trainingMode:
        assert args.cars != '', "The path to the cars training images can't be empty"
        assert args.notcars != '', "The path to the NOT cars training images can't be empty"

    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = False  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    img_height = args.imgHeight
    img_width = args.imgWidth
    x_start_stop = [0, img_width]
    y_start_stop = [int(img_height / 2), img_height]  # Min and max in y to search in slide_window()
    window_sizes = [64, 96, 128]
    heatmap_threshold = 3
    xy_overlap = (0.75, 0.75)

    # Flag for training the classifier
    TRAIN = args.trainingMode

    # Read in cars and notcars
    cars = glob.glob(args.cars)
    notcars = glob.glob(args.notcars)

    if TRAIN is True:
        car_features = VehicleDetection.extract_features(cars, color_space=color_space,
                                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                                         orient=orient, pix_per_cell=pix_per_cell,
                                                         cell_per_block=cell_per_block,
                                                         hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                         hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_features = VehicleDetection.extract_features(notcars, color_space=color_space,
                                                            spatial_size=spatial_size, hist_bins=hist_bins,
                                                            orient=orient, pix_per_cell=pix_per_cell,
                                                            cell_per_block=cell_per_block,
                                                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                                                            hist_feat=hist_feat, hog_feat=hog_feat)

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scalar = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scalar.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.3, random_state=rand_state)

        print('Using:', orient, 'orientations', pix_per_cell,
              'pixels per cell and', cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        svc = LinearSVC()
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the training time for the SVC
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

        classifier_dict = {'classifier': svc, 'scalar': X_scalar}

        classifier_file = open('classifier.p', 'wb')
        pickle.dump(classifier_dict, classifier_file)

    else:
        classifier_dict = pickle.load(open('classifier.p', 'rb'))
        svc = classifier_dict['classifier']
        X_scalar = classifier_dict['scalar']

    vd = VehicleDetection(window_sizes, x_start_stop, y_start_stop, xy_overlap, svc, X_scalar, color_space,
                          spatial_size, hist_bins,
                          orient, pix_per_cell,
                          cell_per_block,
                          hog_channel, spatial_feat,
                          hist_feat, hog_feat, heatmap_threshold)

    clip = VideoFileClip(args.inputVideoPath)

    output_clip = clip.fl_image(vd.process_frame)

    output_clip.write_videofile(args.outputVideoPath, audio=False)

    print("Done.")


if __name__ == '__main__':
    main()
