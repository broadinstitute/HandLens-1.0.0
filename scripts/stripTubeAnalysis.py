from __future__ import division
import cv2
import numpy as np
import argparse
import scipy
import scipy.stats
import scipy.ndimage
import math
import sklearn
from sklearn import model_selection
from skimage.draw import line, polygon
from sklearn.mixture import GaussianMixture
import glob, os
import json
import imutils
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import curve_fit
import matplotlib as mpl
from scipy import linalg
import numpy as np

# np.random.seed(12321)  # for reproducibility
from keras.models import Model
from keras.layers import Input
import keras.layers
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import h5py
from keras import backend as K
# import utils_multiMNIST as U
import keras.backend.tensorflow_backend as tfback
import tensorflow as tf


def _get_available_gpus():
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


tfback._get_available_gpus = _get_available_gpus


def getPredictions(image_file, tube_coords_json, plotting, plt_hist=False):
    image = cv2.imread(image_file)  # image is loaded as BGR
    tube_coords = json.loads(tube_coords_json)
    f = open(image_file + ".coords.txt", "w")
    f.write(tube_coords_json)
    f.close()
    im_lower_dim = min(image.shape[0], image.shape[1])
    blur_size = np.round(im_lower_dim / 500)
    blur_size = int(blur_size if blur_size % 2 == 1 else blur_size + 1)
    cv2.GaussianBlur(image, (blur_size, blur_size),
                     cv2.BORDER_DEFAULT)
    # resize large images
    resize_factor = 1
    if im_lower_dim > 1000:
        resize_factor = 1000 / im_lower_dim
        image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor,
                           interpolation=cv2.INTER_AREA)
    i = 1
    tube_coords = np.asarray(tube_coords) * resize_factor
    strip_count = len(tube_coords) - 1
    # Filter the image to enhance various features
    # image = applyClahetoRGB(image, cv2.COLOR_BAYER_BG2RGB)  # Increase contrast to the image

    unstandardized_scores = [None] * strip_count
    unstandardized_scores_medians = [None] * strip_count
    tmp = image.copy()
    tmp_filtered = image.copy()
    sig_dists = []
    sig_coeffs = []

    # iterate over the tubes
    subimages = []
    for i in range(0, strip_count):
        tube_width = int(((tube_coords[i][2] - tube_coords[i][0]) ** 2 + (
                tube_coords[i][3] - tube_coords[i][1]) ** 2) ** (1 / 2))
        # let's get background intensity so we can normalize the signal from the fluorescent liquid
        box_bg = np.zeros((5, 2))
        box_bg[0] = np.asarray([tube_coords[i][0], tube_coords[i][1]])  # top right
        box_bg[1] = np.asarray([tube_coords[i + 1][0], tube_coords[i + 1][1]])  # bottom right
        box_bg[2] = np.asarray(extend_line(tube_coords[i + 1][2], tube_coords[i + 1][3],
                                           tube_coords[i + 1][0], tube_coords[i + 1][1],
                                           tube_width / 2.5))  # bottom left
        box_bg[3] = np.asarray(extend_line(tube_coords[i][2], tube_coords[i][3], tube_coords[i][0],
                                           tube_coords[i][1], tube_width / 2.5))  # top left
        box_bg[4] = np.asarray([tube_coords[i][0], tube_coords[i][1]])  # top right
        rr_bg, cc_bg = polygon(box_bg[:, 0], box_bg[:, 1])
        mask_background = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask_background[cc_bg, rr_bg] = 1
        subimage_bg = cv2.bitwise_and(image, image, mask=mask_background)
        # subimage_bg[:, :, 0] = np.zeros([subimage_bg.shape[0], subimage_bg.shape[1]])
        bkgd_red = np.median(
            subimage_bg[cc_bg, rr_bg, 2])  # (np.sum(subimage_bg[:, :, 2])) / np.sum(mask)
        bkgd_grn = np.median(
            subimage_bg[cc_bg, rr_bg, 1])  # (np.sum(subimage_bg[:, :, 1])) / np.sum(mask)
        bkgd_blu = np.median(
            subimage_bg[cc_bg, rr_bg, 0])  # (np.sum(subimage_bg[:, :, 0])) / np.sum(mask)
        b_bg, g_bg, r_bg = cv2.split(subimage_bg)
        hist_begin = 1
        hist_end = 255
        hist_bg, edges_bg = np.histogram(g_bg.ravel(), hist_end - hist_begin,
                                         [hist_begin, hist_end])
        edges_bg = (edges_bg[:-1] + edges_bg[1:]) / 2
        # Sometimes we see bright blue/white artifacts in the image. We have to remove them.
        b_bg, g_bg, r_bg, blue_mask_bg = remove_bright_blues(b_bg, g_bg, r_bg, bkgd_blu, tube_width)
        # blue channel is all noise, so get rid of it:
        b_bg[:, :] = np.zeros([b_bg.shape[0], b_bg.shape[1]])
        # red channel is mostly UTM, so get rid of it:
        r_bg[:, :] = np.zeros([r_bg.shape[0], r_bg.shape[1]])
        if plotting:
            tmp = cv2.drawContours(tmp,
                                   [np.array(box_bg[0:4]).reshape((-1, 1, 2)).astype(np.int32)],
                                   0, (255, 0, 0), 2)

        # now, define a subimage for the signal in the tube
        box = np.zeros((5, 2))
        box[0] = np.asarray([tube_coords[i][0], tube_coords[i][1]])  # top left
        box[1] = np.asarray([tube_coords[i][2], tube_coords[i][3]])  # top right
        box[2] = np.asarray([tube_coords[i + 1][2], tube_coords[i + 1][3]])  # bottom right
        box[3] = np.asarray([tube_coords[i + 1][0], tube_coords[i + 1][1]])  # bottom left
        box[4] = np.asarray([tube_coords[i][0], tube_coords[i][1]])  # top left
        rr, cc = polygon(box[:, 0], box[:, 1])
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask[cc, rr] = 1
        # focus in on the tube liquid's enclosing area
        subimage = cv2.bitwise_and(image, image, mask=mask)
        b, g, r = cv2.split(subimage)

        # Sometimes we see bright blue/white artifacts in the image. We have to remove them.
        b, g, r, blue_mask = remove_bright_blues(b, g, r, bkgd_blu, tube_width)
        # blue channel is all noise, so get rid of it:
        b[:, :] = np.zeros([b.shape[0], b.shape[1]])
        # red channel is mostly UTM, so get rid of it:
        r[:, :] = np.zeros([b.shape[0], b.shape[1]])
        subimage = cv2.merge([b, g, r])

        # subtract away background noise level
        # g_mask = g[:, :] < (bkgd_grn.astype("uint8") + 1)
        # r_mask = r[:, :] < (bkgd_red.astype("uint8") + 1)
        # g -= bkgd_grn.astype("uint8")
        # r -= bkgd_red.astype("uint8")
        # g[g_mask] = 0
        # r[r_mask] = 0
        hist_g, edges_g = np.histogram(g.ravel(), hist_end - hist_begin, [hist_begin, hist_end])
        # g[blue_mask] = int(np.mean(g[cc, rr]))
        # # shift green channel to match with background distribution:
        # green_shift = np.argmax(hist_g) - np.argmax(hist_bg)
        green_shift = 0
        # print("green_shift: {}".format(green_shift))
        # if green_shift > 0:
        #     g_mask = g[:, :] <= (green_shift.astype("uint8"))
        #     g -= green_shift.astype("uint8")
        #     g[g_mask] = 0
        # else:
        #     green_shift *= -1
        #     g_mask = g[:, :] > 255 - (green_shift.astype("uint8"))
        #     g += green_shift.astype("uint8")
        #     g[g_mask] = 255

        # r[blue_mask] = int(np.mean(r[cc, rr])) # red channel is mostly UTM, so get rid of it.
        subimage = cv2.merge([b, g, r])
        # We want to get signal from the part of the tube which contains liquid, and not any other
        # background signal. As such, we model the bottom of the tube as a trapezoid and create a
        # kernel to traverse through the tube's enclosing area to find the portion with the highest
        # signal.
        tube_height = int(((box[3][0] - box[0][0]) ** 2 + (box[3][1] - box[0][1]) ** 2) ** (1 / 2))
        angle = np.rad2deg(np.arctan2(box[0][1] - box[1][1], box[1][0] - box[0][0]))
        kernel = create_kernel(tube_width, tube_height, angle, plotting)
        blurs_green = cv2.filter2D(subimage[:, :, 1], cv2.CV_32F, kernel, anchor=(0, 0))
        blurs_red = cv2.filter2D(subimage[:, :, 2], -1, kernel)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(
            blurs_green)  # + blurs_red) # maxLoc is (x, y) == (col, row)
        max_row = maxLoc[1]
        max_col = maxLoc[0]

        # get list of pixels identified as the tube's signal.
        signal_pxs = []
        for m in range(0, kernel.shape[0]):
            if max_row + m >= g.shape[0]:
                print("Error: please place more space between the strips and the edge of"
                      " the captured image")
                break
            for n in range(0, kernel.shape[1]):
                if max_col + n >= g.shape[1]:
                    print("Error: please place more space between the strips and the edge of"
                          " the captured image")
                    break
                if kernel[m, n] != 0:
                    signal_pxs.append(g[max_row + m, max_col + n])

        hist_sig, edges_sig = np.histogram(signal_pxs, hist_end - hist_begin,
                                           [hist_begin, hist_end])

        edges_sig -= green_shift
        edges_sig = edges_sig[0:-1]
        sig_peak = np.max(hist_sig)
        signal_peak_loc = edges_sig[np.argmax(hist_sig)]
        # Make sure background signal within subimage is callibrated to local background signal
        # Fit gaussian for in-tube signal:
        p0 = sig_peak, signal_peak_loc, 4
        coeff = []
        var_matrix = []
        sig_mean = np.mean(signal_pxs)
        # if the image is really bright, it saturates and our curve fitting fails because the center
        # of the gaussian is above 255. In this case, we just set the peak equal to the mean, and
        # based on previous manual inspection of high signals, sd to 20
        if sig_mean > 215:
            coeff = [sig_peak, sig_mean, 20]
        else:
            coeff, var_matrix = curve_fit(gauss, edges_sig, hist_sig, p0=p0)
        # Get the fitted curve
        hist_signal_fit = gauss(edges_sig, *coeff)
        # plt.plot(edges_sig, hist_sig, label='Test data')
        # plt.plot(edges_sig, hist_signal_fit, label='Fitted data')
        # Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
        # print('Fitted mean = ', coeff[1])
        # print('Fitted standard deviation = ', coeff[2])
        bg_peak = np.max(hist_bg)
        bg_peak_loc = edges_bg[np.argmax(hist_bg)]
        p_bg = bg_peak, bg_peak_loc, 4
        coeff_bg, var_matrix_bg = curve_fit(gauss, edges_bg, hist_bg, p0=p_bg)
        # print('Fitted mean = ', coeff_bg[1])
        # print('Fitted standard deviation = ', coeff_bg[2])
        # Get the fitted curve
        hist_bg_fit = gauss(edges_bg, *coeff_bg)
        sig_dist = []
        for j in range(0, len(hist_bg)):
            sig_dist.extend([edges_sig[j]] * hist_sig[j])
        sig_dist -= coeff_bg[1]
        sig_dists.append(sig_dist)
        sig_coeffs.append(coeff)
        # if plotting and plt_hist:
        #     plt.hist(g_bg[cc_bg, rr_bg].ravel(), bins=40, label="background")
        #     plt.hist(sig_dist, bins=40, label="signal")
        #     plt.legend()
        #     plt.title("tube {}".format(i))
        #     plt.show()

        big_box = np.zeros((5, 2))
        big_box[0] = box_bg[3]  # top left
        big_box[1] = box_bg[2]  # bottom left
        big_box[2] = box[2]  # bottom right
        big_box[3] = box[1]  # top right
        big_box[4] = box_bg[3]

        big_box2 = np.zeros((5, 2))
        max_x = int(np.round(np.max(big_box[:, 0])))
        max_y = int(np.round(np.max(big_box[:, 1])))
        min_x = int(np.round(np.min(big_box[:, 0])))
        min_y = int(np.round(np.min(big_box[:, 1])))
        big_box2 = np.asarray(
            [[max_x, max_y], [max_x, min_y], [min_x, min_y], [min_x, max_y], [max_x, max_y]])
        subimages.append(image[min_y:max_y, min_x:max_x])

        if plotting:
            if i == strip_count - 1:
                tmp = cv2.drawContours(tmp,
                                       [np.array(box[0:4]).reshape((-1, 1, 2)).astype(np.int32)],
                                       0, (0, 255, 255), 1)
            else:
                tmp = cv2.drawContours(tmp,
                                       [np.array(box[0:4]).reshape((-1, 1, 2)).astype(np.int32)],
                                       0, (0, 0, 255), 1)
                tmp = cv2.drawContours(tmp,
                                       [np.array(big_box[0:4]).reshape((-1, 1, 2)).astype(
                                           np.int32)],
                                       0, (255, 255, 0), 1)
                tmp = cv2.drawContours(tmp,
                                       [np.array(big_box2[0:4]).reshape((-1, 1, 2)).astype(
                                           np.int32)],
                                       0, (255, 255, 255), 1)

            if plt_hist:
                fig, ax1 = plt.subplots()
                ax1.hist(g.ravel(), hist_end - hist_begin, [hist_begin, hist_end],
                         log=True, label="subimage")
                ax1.hist(g_bg[cc_bg, rr_bg].ravel(), hist_end - hist_begin, [hist_begin, hist_end],
                         log=True, label="background")
                # get hist of potential signal elements
                ax1.plot(edges_sig, hist_sig, label="signal")
                ax1.set_title('tube {}\n{}'.format(i, image_file.split('\\')[-1]))
                ax1.plot(edges_sig, hist_signal_fit, '--', label='signal fit')
                ax1.set_ylim([0.5, np.max(hist_bg) * 2])
                plt.legend()
                plt.show()

        unstandardized_scores[i] = coeff[1] - coeff_bg[1]  # np.median(signal_pxs) - coeff_bg[1]
        unstandardized_scores_medians[i] = np.median(signal_pxs) - coeff_bg[1]

        # unstandardized_scores[i] = abs(maxVal - bkgd_grn.astype("uint8"))
        # print("maxVal: {}".format(maxVal))
        # print("bkgd_grn: {}".format(bkgd_grn.astype("uint8")))
        # print("unstandardized_scores: {}".format(unstandardized_scores[i]))
        # print()
        # unstandardized_scores[i] = maxVal

    for i in range(0, len(subimages)):
        subimage = subimages[i]
        if subimage.shape[0] < subimage.shape[1]:
            subimage = scipy.ndimage.rotate(subimage, -90)
        subimage = subimage[2:-2, 2:-2]
        subimages[i] = cv2.resize(subimage, (31, 47))
    if plotting:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        plt.title('{}'.format(image_file.split('\\')[-1]))
        plt.show()
    # print("unstandardized_scores: {}".format(unstandardized_scores))
    for i in range(0, len(sig_dists)):
        continue
        # if plotting and plt_hist:
        #     plt.hist(sig_dists[i], bins=40, label="tube {} signal".format(i))
        #     plt.hist(sig_dists[-1], bins=40, label="control signal")
        #     plt.legend()
        #     plt.show()
    # final_score = [unstandardized_score / unstandardized_scores[-1]
    #                for unstandardized_score in unstandardized_scores]
    final_score = []
    final_score_medians = []
    final_score = list((unstandardized_score - unstandardized_scores[-1]) / sig_coeffs[-1][2]
                       for unstandardized_score in unstandardized_scores)
    final_score_medians = list(
        (unstandardized_score_median - unstandardized_scores[-1]) / sig_coeffs[-1][2]
        for unstandardized_score_median in unstandardized_scores_medians)
    f = open(image_file + ".scores.txt", "w")
    f.write(json.dumps(final_score))
    f.close()

    return final_score, final_score_medians, subimages


def extend_line(x1, y1, x2, y2, length):
    lenAB = math.sqrt(pow(x1 - x2, 2.0) + pow(y1 - y2, 2.0))
    x = x2 + (x2 - x1) / lenAB * length
    y = y2 + (y2 - y1) / lenAB * length
    return x, y


def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def remove_bright_blues(b, g, r, bkgd_blu, tube_width):
    bf = b.astype(float) + .001
    rf = r.astype(float) + .001
    gf = g.astype(float) + .001
    blue_cutoff = 2 * bkgd_blu
    blue_mask1 = bf[:, :] > blue_cutoff
    blue_mask2 = bf[:, :] / gf[:, :] > 1
    blue_mask = np.logical_and(blue_mask1, blue_mask2)
    pixel_threshold = (tube_width / 10) ** 2
    # if np.sum(blue_mask) > pixel_threshold:
    #     # be more stringent if we think we've detected a bright blue/white artifact
    #     blue_cutoff = bkgd_blu * 1.5
    #     blue_mask = b[:, :] > blue_cutoff
    g[blue_mask] = 0
    r[blue_mask] = 0

    return b, g, r, blue_mask


def create_kernel(tube_width, tube_height, angle, plotting):
    """
    :return: a trapezoidal kernel
    kernel_height = int(tube_height
    """
    kernel_width = tube_width
    kernel_height = tube_height
    kernel = np.zeros((kernel_height, kernel_width), np.float32)
    trap_height_large = int(24 * tube_height / 75)
    trap_height_small = int(11 * tube_height / 75)
    trap_length = int(33 * tube_width / 75)
    trapezoid = np.zeros((4, 2))
    trapezoid[0] = np.asarray([kernel_height / 2 - trap_height_small / 2, 0])  # top left
    trapezoid[1] = np.asarray([kernel_height / 2 - trap_height_large / 2, trap_length])  # top right
    trapezoid[2] = np.asarray(
        [kernel_height / 2 + trap_height_large / 2, trap_length])  # bottom right
    trapezoid[3] = np.asarray([kernel_height / 2 + trap_height_small / 2, 0])  # bottom left
    rr, cc = polygon(trapezoid[:, 0], trapezoid[:, 1], kernel.shape)
    kernel[rr, cc] = 1
    kernel = kernel / cv2.sumElems(kernel)[0]
    kernel = imutils.rotate_bound(kernel, -1 * angle)
    # if plotting:
    #     plt.figure()
    #     plt.imshow(kernel)
    #     plt.show()

    return kernel


# from https://stackoverflow.com/a/37123933
def check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape)  # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    if p1[0] == p2[0]:
        max_col_idx = (idxs[0] - p1[0]) * idxs.shape[1]
        sign = np.sign(p2[1] - p1[1])
    else:
        max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) + p1[1]
        sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign


# from https://stackoverflow.com/a/37123933
def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros

    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k - 1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1 / (shape[0] * shape[1])

    return base_array


def applyClahetoRGB(bgr_imb):
    lab = cv2.cvtColor(bgr_imb, cv2.COLOR_BGR2LAB)
    # Split lab image to different channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Convert image from LAB Color model to RGB model
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def run_analysis(file, tube_coords, threshold, plotting=True, plt_hist=False):
    fs, fs_m, subimages = getPredictions(file, tube_coords, plotting, plt_hist)
    calls = [1 if fs[i] > threshold and fs_m[i] > threshold else 0 for i in range(0, len(fs))]
    # print(json.dumps({"calls": calls, "final_scores": fs, "final_scores_median": fs_m}))
    return calls


def main():
    parser = argparse.ArgumentParser('Read strip tubes')
    parser.add_argument('--image_file', required=False)
    parser.add_argument('--tubeCoords', required=False)
    parser.add_argument('--plotting', help="Enable plotting", action='store_true')
    args = parser.parse_args()
    threshold = 1.0

    image_sets = []
    ys_binary = []
    ys_regression = []
    train_threshold = False
    if train_threshold:
        thresholds = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    elif False:
        # load images
        image_sets = np.load("image_sets.npy", allow_pickle=True)
        images = []
        regression_values = []
        binary_values = []

        for strips in image_sets:
            for tube in strips:
                images.append(tube)
        images = np.asarray(images)

        regression_sets = np.load("ys_regression.npy", allow_pickle=True)
        for li in regression_sets:
            regression_values.extend(li)
        regression_values = np.asarray(regression_values)

        binary_sets = np.load("ys_binary.npy", allow_pickle=True)
        for li in binary_sets:
            binary_values.extend(li)
        binary_values = np.asarray(binary_values)
        ct_pos = 0
        ct_neg = 0
        for i in range(0, len(binary_values)):
            if binary_values[i]:
                ct_pos += 1
            else:
                ct_neg += 1
        print(ct_pos)
        print(ct_neg)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(images,
                                                                                    binary_values,
                                                                                    test_size=0.2,
                                                                                    random_state=42)
        X_train_exp = X_train[:, 0:47, :, :]
        X_train_ctrl = X_train[:, 47:, :, :]

        X_test_exp = X_test[:, 0:47, :, :]
        X_test_ctrl = X_test[:, 47:, :, :]

        nb_epoch = 15
        x1 = Input(shape=(47, 31, 3))
        conv1 = Conv2D(8, (5, 5), activation='relu', data_format='channels_last')(x1)
        mp1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1)
        conv2 = Conv2D(16, (3, 3), activation='relu')(mp1)
        mp2 = MaxPooling2D((2, 2), strides=(1, 1))(conv2)
        flat = Flatten()(mp2)
        x1b = Input(shape=(47, 31, 3))
        conv1b = Conv2D(8, (5, 5), activation='relu', data_format='channels_last')(x1b)
        mp1b = MaxPooling2D((2, 2), strides=(2, 2))(conv1b)
        conv2b = Conv2D(16, (3, 3), activation='relu')(mp1b)
        mp2b = MaxPooling2D((2, 2), strides=(1, 1))(conv2b)
        flatb = Flatten()(mp2b)
        x = keras.layers.concatenate([flat, flatb])
        dense = Dense(128, activation='relu')(x)
        you_should_stay_in_school = Dropout(0.5)(dense)
        predictions_1 = Dense(1, activation='sigmoid')(you_should_stay_in_school)
        model = Model(inputs=[x1, x1b], outputs=[predictions_1])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # fit model:
        print(X_train.shape)
        model.fit([X_train_exp, X_train_ctrl], y_train, nb_epoch=nb_epoch, verbose=1)
        # classifications = [True if (ypred[i] > 1.5 and y_test[i] > 1.5) or
        #                            (ypred[i] > 1.5 and y_test[i] > 1.5) else False for i in
        #                    range(0, len(ypred))]
        # positives = 0
        # for res in classifications:
        #     if res:
        #         positives += 1

        objective_score = model.evaluate([X_test_exp, X_test_ctrl], y_test)
        print('Evaluation on test set:', dict(zip(model.metrics_names, objective_score)))

    elif True:
        truth_files = glob.glob(
            r'C:\Users\Sameed\Documents\Educational\PhD\Pardis\SHERLOCK-reader\covid\jon_pictures\uploads\*truths.txt')
        ta = glob.glob(
            r'C:\Users\Sameed\Documents\Educational\PhD\Pardis\SHERLOCK-reader\covid\jon_pictures\uploads\*truths.txt')
        kabadra = 0
        kabadra_limit = 200
        correct = 0
        incorrect = 0
        for file in glob.glob(
                r'C:\Users\Sameed\Documents\Educational\PhD\Pardis\SHERLOCK-reader\covid\jon_pictures\uploads\*jpg'):
            # if "IMG_chu113_90min_saliva" not in file:
            #     continue
            reqs = [file + ".truths.txt", file + ".truths-analytical.txt", file + ".coords.txt"]
            if not os.path.exists(reqs[0]):
                print("scooby")
                continue
            if not os.path.exists(reqs[1]):
                print("doo")
                continue
            if not os.path.exists(reqs[2]):
                print("boo")
                continue
            print(file)
            tube_coords = None
            with open(file + ".coords.txt") as f:
                for line in f:  # there should only be one line in file f
                    tube_coords = line
            calls = run_analysis(file, tube_coords, threshold, plotting=True, plt_hist=True)
            f = open(reqs[0])
            binary_data = [True if 't' == d else False for d in json.load(f)]

            for i in range(0, len(calls) - 1):
                if calls[i] == binary_data:
                    correct += 1
                else:
                    incorrect += 1
        print(correct)
        print(incorrect)

    else:
        final_scores = run_analysis(args.image_file, args.tubeCoords, threshold, args.plotting)


if __name__ == '__main__':
    main()
