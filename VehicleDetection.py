import pickle
from collections import deque

import cv2
from scipy.ndimage.measurements import label

from FeatureExtraction import *


class VehicleDetector:
    def __init__(self, model_param_files):
        # Loading Model Parameters
        with open(model_param_files, 'rb') as pfile:
            pickle_data = pickle.load(pfile)
            for key in pickle_data:
                exec("self." + key + "= pickle_data['" + str(key) + "']")
            del pickle_data

        # Current HeatMap
        self.heatmap = None

        # Heat Image for the Last Three Frames
        self.heat_images = deque(maxlen=3)

        # Current Frame Count
        self.frame_count = 0
        self.full_frame_processing_interval = 10

        # Xstart
        self.xstart = 600

        # Various Scales
        self.ystart_ystop_scale = [(380, 480, 1), (400, 600, 1.5), (500, 700, 2.5), (400, 464, 1.0), (416, 480, 1), (400, 496, 1.5), (432, 532, 1.5), (400, 528, 2.0), (430, 560, 2.0), (400, 596, 3.5), (464, 660, 3.5)]

        # Kernal For Dilation
        self.kernel = np.ones((50, 50))

        # Threshold for Heatmap
        self.threshold = 0

        self.labeled_img = None

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, rect_return=False, show_all_rectangles=False):

        X_scaler = self.X_scaler
        orient = self.orient
        pix_per_cell = self.pix_per_cell
        cell_per_block = self.cell_per_block
        spatial_size = self.spatial_size
        hist_bins = self.hist_bins
        svc = self.svc
        color_space = self.color_space
        hog_channel = self.hog_channel

        color_space = 'YCrCb'
        spatial_size = (32, 32)
        hist_bins = 32
        orient = 11
        pix_per_cell = 16
        cell_per_block = 2
        hog_channel = 'ALL'
        spatial_feat = True
        hist_feat = True
        hog_feat = True

        draw_img = np.copy(img)
        img = img.astype(np.float32) / 255

        drawboxes = []

        for (ystart, ystop, scale) in self.ystart_ystop_scale:

            img_tosearch = img[ystart:ystop, :, :]
            ctrans_tosearch = convert_color(img_tosearch, conv='RGB2' + color_space)
            if scale != 1:
                imshape = ctrans_tosearch.shape
                ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

            ch1 = ctrans_tosearch[:, :, 0]
            ch2 = ctrans_tosearch[:, :, 1]
            ch3 = ctrans_tosearch[:, :, 2]

            # Define blocks and steps as above
            nxblocks = (ch1.shape[1] // pix_per_cell) + 1
            nyblocks = (ch1.shape[0] // pix_per_cell) + 1
            nfeat_per_block = orient * cell_per_block ** 2
            # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
            window = 64
            nblocks_per_window = (window // pix_per_cell) - 1
            cells_per_step = 2  # Instead of overlap, define how many cells to step
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step

            # Compute individual channel HOG features for the entire image
            if hog_channel == 'ALL':
                hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
                hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
                hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
            else:
                hog1 = get_hog_features(hog_channel, orient, pix_per_cell, cell_per_block, feature_vec=False)

            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb * cells_per_step
                    xpos = xb * cells_per_step
                    # Extract HOG for this patch

                    if hog_channel == 'ALL':
                        hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                    else:
                        hog_features = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

                    xleft = xpos * pix_per_cell
                    ytop = ypos * pix_per_cell

                    # Extract the image patch
                    subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                    # Get color features
                    spatial_features = bin_spatial(subimg, size=spatial_size)
                    hist_features = color_hist(subimg, nbins=hist_bins)

                    # Scale features and make a prediction
                    test_features = X_scaler.transform(
                        np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                    # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                    test_prediction = svc.predict(test_features)

                    if test_prediction == 1 or show_all_rectangles:
                        xbox_left = np.int(xleft * scale)
                        ytop_draw = np.int(ytop * scale)
                        win_draw = np.int(window * scale)
                        # print(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                        drawboxes.append(
                            ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                        #cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                     # (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
            # result.append((10, 20))

                        # Add heat to each box in box list
        self.add_heatmap_and_threshold(draw_img, drawboxes, self.threshold)

        # Find final boxes from heatmap using label function
        t_heatmap = self.heatmap
        labels = label(t_heatmap)
        label_img = np.copy(draw_img)
        VehicleDetector.draw_labeled_bboxes(label_img, labels)

        return drawboxes if rect_return else label_img


############



    def add_heatmap_and_threshold(self, draw_img, bbox_list, threshold):
        heatmap = np.zeros_like(draw_img[:, :, 0])

        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        self.heat_images.append(heatmap)
        self.heatmap = np.sum(np.array(self.heat_images), axis=0)
        self.heatmap[self.heatmap <= threshold] = 0

    @staticmethod
    def draw_labeled_bboxes(img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)