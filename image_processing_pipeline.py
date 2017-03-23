import numpy as np
import glob
import time
import os
import pickle
import pdb
from skimage.feature import hog
from scipy.ndimage.measurements import label

from feature_generation import *

## paramters for vehicle shearaching

# load classifier model and data normalizing factors
with open('data/model.p', 'rb') as f:
    model, X_scaler = pickle.load(f)
# image frame used for searching vehicle in the image
im_size=(64, 64)
# area of image in which vehicle is searched
ystart=400
ystop=650
# image resizing factor
scale=1
# HOG Parameters
orient=9
pix_per_cell=8
cell_per_block=2
# Spacial Image Parameters
spatial_size=(32, 32)
# Color Histogram Parameters
hist_bins=32
# Temporal frame
frame_memory=5
threshold=3
frame_buffer=[]

def find_cars(im, DEBUG = False):
    img = im.astype(np.float32) / 255
    heatmap = np.zeros_like(img)

    img_tosearch = img[ystart:ystop, ...]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    if scale != 1:
        h, w, ch = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (w // scale, h // scale))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]
    if DEBUG:
        print('channel 1: ', ch1.shape[1])
        print('channe2 1: ', ch2.shape[0])
        print('channe3 1: ', ch3.shape[0])
    # Define blocks and steps as above
    if DEBUG:
        print('output values: ',  (ch1.shape[1] // pix_per_cell) )
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1

    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    if DEBUG:
        print('number of steps : ',  nxsteps, ' and type: ', type(nxsteps) )
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], im_size)

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack(( spatial_features, hist_features, hog_features))).reshape(1, -1)
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = model.predict(test_features)
            if 0:
                print('test prediction: ', test_prediction)
                
            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                top_left = (xbox_left, ytop_draw + ystart)
                bottom_right = (xbox_left + win_draw, ytop_draw + win_draw + ystart)
                if DEBUG:
                    print('area of prediction: ', xbox_left, ytop_draw, win_draw, top_left, bottom_right )
                add_heat(heatmap, (top_left, bottom_right))

    add_to_buffer(heatmap)
    avg_heatmap = get_heatmap_from_buffer()
    labels = label(avg_heatmap)
    return draw_labeled_bounding_boxes(im, labels)

def add_to_buffer(heat_map):
    if len(frame_buffer) == frame_memory:
        frame_buffer.pop(0)
    frame_buffer.append(heat_map)
    return

def get_heatmap_from_buffer():
    heatmap_sum = np.zeros_like(frame_buffer[0])
    for heatmap in frame_buffer:
        heatmap_sum = np.add(heatmap_sum, heatmap)
    heatmap_sum[heatmap_sum < threshold] = 0
    return heatmap_sum


def add_heat(heatmap, window):
    xstart, ystart = window[0]
    xend, yend = window[1]
    heatmap[ystart:yend, xstart:xend, ...] += 1
    return heatmap

def draw_labeled_bounding_boxes(im, labels):
    cpy = np.copy(im)
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        bbox = ((np.min(nonzero_x), np.min(nonzero_y)), (np.max(nonzero_x), np.max(nonzero_y)))
        cv2.rectangle(cpy, bbox[0], bbox[1], (0, 0, 255), 6)
    return cpy