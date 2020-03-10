from __future__ import division
import statistics
import cv2
import imutils
import math
import numpy as np
import argparse
import sys
import matplotlib
import os

if __name__ == '__main__':
    from strip_analysis import MaxDetector
    from strip_analysis import correct_input_image
    from strip_analysis import convert_image_to_linear_signal
else:
    from .strip_analysis import MaxDetector
    from .strip_analysis import correct_input_image
    from .strip_analysis import convert_image_to_linear_signal

import matplotlib.pyplot as plt


def makeOrderedBox(rect):
    """
    Return a 4-element tuple representing the corners of a box:
        idx 0 = top left corner
        idx 1 = top right corner
        idx 2 = bottom right corner
        idx 3 = botton left corner
    """
    box0 = cv2.boxPoints(rect)
    box0 = np.int0(box0)

    xval = [pt[0] for pt in box0]
    yval = [pt[1] for pt in box0]

    x0 = np.mean(xval)
    y0 = np.mean(yval)

    angles = []
    for i in range(0, len(box0)):
        xi = box0[i][0]
        yi = box0[i][1]
        x = xi - x0
        y = yi - y0
        a = np.arctan2(y, x)
        val = [a, i]
        angles += [val]

    angles.sort(key=lambda val: val[0], reverse=False)
    box = np.array([box0[val[1]] for val in angles])

    return box


def boxMinX(box):
    return min([pt[0] for pt in box])


def boxMaxX(box):
    return max([pt[0] for pt in box])


def boxMinY(box):
    return min([pt[1] for pt in box])


def boxMaxY(box):
    return max([pt[1] for pt in box])


def boxArea(box):
    x0 = np.mean([pt[0] for pt in box])
    y0 = np.mean([pt[1] for pt in box])
    p0 = np.array([x0, y0])

    area = 0
    n = len(box)
    for i in range(0, n):
        p1 = box[i]
        if i < n - 1:
            p2 = box[i + 1]
        else:
            p2 = box[0]

        # Heron's Formula
        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p0)
        c = np.linalg.norm(p1 - p2)
        s = (a + b + c) / 2
        triarea = np.sqrt(s * (s - a) * (s - b) * (s - c))

        area += triarea

    return area


def rectArea(rect):
    return rect[1][0] * rect[1][1]


def pointDistance(p1, p2):
    """
    Given two poiints, each represented by a tuple (x1, y1), calculate the eucalidian distance
    between them.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


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


def getmin(data):
    half = int(data.shape[0] / 2)
    top = data.shape[0]
    values = np.array([])
    for i in range(half, top):
        values = np.append(values, np.mean(data[i]))
    m = np.argmin(values)
    sd = np.std(values)
    return values[m], sd, m + half


# Naive prediction using threshold obtained from set of controls and low concentration samples
def predict(data, threshold, LODStandardDeviation = 29):
    m, sd, min_pos = getmin(data)
    baseline = np.mean(data[400: 550])

    signal_height = baseline - m

    f = ((threshold + LODStandardDeviation) - signal_height) / (LODStandardDeviation*2)
    if f < 0: f = 0
    if 1 < f: f = 1
    score = 1 - f
    return score


def intersectLines(pt1, pt2, ptA, ptB):
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """
    # From:
    # https://www.cs.hmc.edu/ACM/lectures/intersections.html

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1;
    x2, y2 = pt2
    dx1 = x2 - x1;
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA;
    xB, yB = ptB;
    dx = xB - x;
    dy = yB - y;

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE: return (0, 0, 0, 0, 0)

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return (xi, yi, 1, r, s)


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def rotate_image(img, center, angle, width, height):
    shape = (img.shape[1], img.shape[0])  # (length, height)

    matrix = cv2.getRotationMatrix2D((center[0], center[1]), angle, 1)
    rotated = cv2.warpAffine(img, matrix, shape)

    x = int(center[0] - width / 2)
    y = int(center[1] - height / 2)

    cropped = rotated[y:y + height, x:x + width]

    return cropped


def getPredictions(filename, stripPixelArea, plotting=False):
    stripWidthRelative = 7
    stripHeightRelative = 48

    # The maximum and minimum allowed ratios of the sides of the green box
    maxGreenBoxRatio = 140.0 / 528
    minGreenBoxRatio = 102.0 / 578
    maxGreenBoxArea = stripPixelArea * 90 / 300
    minGreenBoxArea = stripPixelArea * 35 / 300

    # The maximum allowed ratio of the sides of the final deteted strip
    maxStripBoxRatio = stripWidthRelative / stripHeightRelative

    # The maximum and minimum allowed ratios of the sides of the box defined by the red arrows
    maxRedBoxRatio = 42.0 / 91
    minRedBoxRatio = 14.0 / 104
    maxRedBoxArea = stripPixelArea * 75 / 300
    minRedBoxArea = stripPixelArea * 20 / 300

    # The maximum and minimum allowed ratios of the areas of green boxes and red arrow-bounding boxes
    maxRedGreenBoxRatio = 3500.0 / 4000
    minRedGreenBoxRatio = 2000.0 / 4500

    # Sensitive area of the strip
    stripWidth = 200
    stripHeight = 2000
    stripHoldY = 900
    # maxPeakAlign = 50

    # Percentage of the margins to be removed
    marginFraction = 0.2

    # LOD Standard deviation obtained from low concentration samples
    LODStandardDeviation = 29
    ThresholdFactor = 0.5

    # Red is at the beginning/end of the hue range, so it covers the [0-15] and the [170, 180]
    # (hue in OpenCV varies  between 0 and 180 degrees)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([13, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Green is between 20 and 90 (these ranges can be adjusted)
    lower_green = np.array([20, 50, 50])
    upper_green = np.array([83, 255, 255])

    # We can also use a large color range to encapsulate both red and green:
    lower_redgreen1 = np.array([0, 50, 50])
    upper_redgreen1 = np.array([83, 255, 255])
    lower_redgreen2 = np.array([170, 50, 50])
    upper_redgreen2 = np.array([180, 255, 255])

    image = cv2.imread(filename)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # First, convert the image to HSV color space, which makes the color detection straightforward
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # These strips have red arrows on a green background, so we define two masks, one for red and the
    # other for green

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = red_mask1 + red_mask2
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    redgreen_mask1 = cv2.inRange(hsv, lower_redgreen1, upper_redgreen1)
    redgreen_mask2 = cv2.inRange(hsv, lower_redgreen2, upper_redgreen2)
    mask = green_mask # redgreen_mask1 + redgreen_mask2

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    if plotting:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
        plt.show()

    '''adjust brightness'''
    # resize the image, and then create a kernel
    k_factor = math.sqrt(stripPixelArea / (stripWidthRelative * stripHeightRelative))
    resized_img = cv2.resize(image, (0, 0), fx=(1 / k_factor), fy=(1 / k_factor))
    grey_resized = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((stripHeightRelative, stripWidthRelative), np.float32) / (
            stripWidthRelative * stripHeightRelative)
    dst = cv2.filter2D(grey_resized, -1, kernel)

    if plotting:
        plt.imshow(cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB))
        plt.show()
    _, maxVal, _, _ = cv2.minMaxLoc(dst)

    minStripThreshold = 0.6 * maxVal

    '''
    # Processing Step 2: finding green box candidates
    # Stage 1:
    #     Given a masked version of the image which includes red to green hues,
    #     find all green contours
    # Stage 2:
    #     Declare a green contour to be a green_rect_candidate to be further 
    #     analyzed if the size is in the expected range
    '''

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Create a copy of the original image to draw the bounding boxes on
    tmp = image.copy()

    green_box_candidates = []
    green_rect_candidates = []

    green_box_candidates = []
    green_rect_candidates = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] == 0: continue

        rect = cv2.minAreaRect(c)
        box = makeOrderedBox(rect)

        # rect is a tuple with the following details:
        #    ( (x,y) of middle of top side of rect,  (width,height),  angle of rotation )
        # As such, we can easily get the ratio of the sides of the detected boxes; note
        # that because for our purpose the sides are categorized into width and height
        # arbitrarily, we take the ratio of the sides as the lesser ratio:
        greenBoxRatio = min(rect[1][0] / rect[1][1], rect[1][1] / rect[1][0])
        area = boxArea(box)

        green_box_candidates += [box[0:]]
        green_rect_candidates += [rect]

    green_box_candidates.sort(key=lambda box: boxArea(box), reverse=True)
    green_rect_candidates.sort(key=lambda rect: rectArea(rect), reverse=True)
    green_rect_len = max(green_rect_candidates[1][1][0],
                         green_rect_candidates[1][1][1])

    # In some cases, the red arrows split the green boxes into two.
    # We fix this by merging two boxes if the bottom left and top left corners of the respective
    # boxes, and the  right and top right corners of the respective boxes are within a threshold
    # distance of  each other.
    green_boxes = []
    green_rects = []
    green_box_candidates.sort(key=lambda item: item[1][1])

    distance_threshold = green_rect_len * 0.20
    merged_boxes = {}
    tmp = image.copy()
    for i in range(0, len(green_box_candidates)):
        upper_box = green_box_candidates[i]
        if i in merged_boxes:
            continue
        current_merged_upper_box = upper_box
        for j in range(i, len(green_box_candidates)):
            lower_box = green_box_candidates[j]

            if pointDistance(current_merged_upper_box[3], lower_box[0]) < distance_threshold and \
                    pointDistance(current_merged_upper_box[2], lower_box[1]) < distance_threshold:
                # Sometimes the arrows get detected in this step as false boxes -- to filter for
                # this, we make sure that boxes can only be concatenated if they have similar width:
                if pointDistance(current_merged_upper_box[3], current_merged_upper_box[2]) < \
                        pointDistance(lower_box[0], lower_box[1]) * 1.5:
                    current_merged_upper_box = np.array([current_merged_upper_box[0],
                                                         current_merged_upper_box[1],
                                                         lower_box[2], lower_box[3]])
                    if minGreenBoxArea < boxArea(current_merged_upper_box):
                        merged_boxes[j] = True

        else:
            stripBoxRatio = pointDistance(current_merged_upper_box[0],
                                          current_merged_upper_box[1]) / \
                            pointDistance(current_merged_upper_box[0],
                                          current_merged_upper_box[3])
            if minGreenBoxArea < boxArea(current_merged_upper_box):  # and \
                #                 stripBoxRatio < maxStripBoxRatio:
                green_boxes.append(current_merged_upper_box)

    # Create a copy of the original image to draw the bounding boxes on
    tmp = image.copy()
    for box in green_boxes:
        tmp = cv2.drawContours(tmp, [box], 0, (0, 0, 255), 10)
        tmp = cv2.circle(tmp, (box[0][0], box[0][1]), 20, (255, 0, 0), -1)

    if plotting:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        plt.show()

    green_boxes.sort(key=lambda box: boxArea(box), reverse=True)
    green_rects.sort(key=lambda rect: rectArea(rect), reverse=True)

    green_rect_len = pointDistance(green_boxes[0][3], green_boxes[0][0])
    greenBoxArea = boxArea(green_boxes[1])

    '''
    # Processing Step 3: binary thresholding of the entire image to extract the top part of the
    strips
    '''
    ret, thresh = cv2.threshold(image, minStripThreshold, 255, cv2.THRESH_BINARY)
    blur_size = thresh.shape[0]//150
    blur_size = blur_size + 1 if blur_size % 2 == 0 else blur_size # medianBlur size must be odd
    thresh = cv2.medianBlur(thresh,  blur_size)

    if plotting:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
        plt.show()

    '''
    # Processing Step 4: detect boundary boxes for the top of the strips
    '''

    grayscale = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Create a copy of the original image to draw the bounding boxes on
    tmp = image.copy()

    # Minimum area of the top strip boxes
    minTopBoxArea = 0.2 * stripPixelArea
    # Maximum of the top strip boxes
    maxTopBoxArea = 3 * stripPixelArea

    top_box_candidates = []
    top_rects_candidates = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] == 0: continue

        rect = cv2.minAreaRect(c)
        box = makeOrderedBox(rect)

        area = boxArea(box)
        if area < minTopBoxArea / 15 or maxTopBoxArea < area:
            continue

        top_box_candidates += [box]
        top_rects_candidates += [rect]

        tmp = cv2.drawContours(tmp, [box], 0, (0, 0, 255), 10)
        tmp = cv2.circle(tmp, (box[0][0], box[0][1]), 20, (255, 0, 0), -1)

    if plotting:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        plt.show()

    '''
    In some cases, the control or pos signal is so strong that a continuous box gets split up into
    two boxes. We fix this by merging two boxes if the bottom left and top left corners of the
    respective boxes, and the  right and top right corners of the respective boxes are within a
    threshold distance of each other. 
    '''
    top_boxes = []
    top_box_candidates.sort(key=lambda item: item[1][1])
    distance_threshold = green_rect_len * 0.15
    merged_boxes = {}
    tmp = image.copy()
    strip_boxes = []
    for i in range(0, len(top_box_candidates)):
        upper_box = top_box_candidates[i]
        if i in merged_boxes:
            continue
        current_merged_upper_box = upper_box
        for j in range(i, len(top_box_candidates)):
            lower_box = top_box_candidates[j]
            if pointDistance(current_merged_upper_box[3], lower_box[0]) < distance_threshold and \
                    pointDistance(current_merged_upper_box[2], lower_box[1]) < distance_threshold:
                # Sometimes the arrows get detected in this step as false boxes -- to filter for this,
                # we make sure that boxes can only be concatenated if they have similar width:
                if pointDistance(current_merged_upper_box[3], current_merged_upper_box[2]) < \
                        pointDistance(lower_box[0], lower_box[1]) * 1.5:
                    current_merged_upper_box = np.array([current_merged_upper_box[0],
                                                         current_merged_upper_box[1],
                                                         lower_box[2], lower_box[3]])

                    if minTopBoxArea < boxArea(current_merged_upper_box):
                        merged_boxes[j] = True
        else:
            stripBoxRatio = pointDistance(current_merged_upper_box[0],
                                          current_merged_upper_box[1]) / \
                            pointDistance(current_merged_upper_box[0],
                                          current_merged_upper_box[3])
            if minTopBoxArea < boxArea(current_merged_upper_box) and \
                    stripBoxRatio < maxStripBoxRatio:
                top_boxes.append(current_merged_upper_box)

    for box in top_boxes:
        tmp = cv2.drawContours(tmp, [box], 0, (0, 0, 255), 5)
        tmp = cv2.circle(tmp, (box[0][0], box[0][1]), 20, (255, 0, 0), -1)

    if plotting:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        plt.show()

    '''
    # Processing Step 5: construct the boxes that enclose the sensitive strip area
    '''

    # First, order the top boxes from left to right
    top_boxes.sort(key=lambda box: box[0][0], reverse=False)
    # Find the green boxes which bound arrows, and then
    # order them left to right
    green_boxes.sort(key=lambda box: boxArea(box), reverse=True)
    green_boxes = green_boxes[0:len(top_boxes)]
    green_boxes.sort(key=lambda box: box[0][0], reverse=False)

    # Create a copy of the original image to draw the bounding boxes on
    tmp = image.copy()

    num_boxes = len(top_boxes)

    # strip_boxes = []

    for i in range(0, num_boxes):
        tbox = top_boxes[i]
        rbox = green_boxes[i]

        # The corners are expected to be received in the following order:
        # 0 = botton left corner
        # 1 = top left corner
        # 2 = top right corner
        # 3 = bottom right corner

        tp0, tp1, tp2, tp3 = tbox[3], tbox[0], tbox[1], tbox[2]
        rp0, rp1, rp2, rp3 = rbox[3], rbox[0], rbox[1], rbox[2]

        # The intersection of the lines defining the sides of the strip (tp1-tp0 and tp2-tp3)
        # with the bottom edge of the green box defines the bottom corners of the area of
        # interest
        res1 = intersection(line(tp1, tp0), line(rp0, rp3))
        res2 = intersection(line(tp2, tp3), line(rp0, rp3))

        assert (res1 != False and res2 != False), "Top and center boxes are not intersecting"

        p1 = np.array([int(round(res1[0])), int(round(res1[1]))])
        p2 = np.array([int(round(res2[0])), int(round(res2[1]))])

        sbox = np.array([p1, tp1, tp2, p2])
        # #     sbox = np.array([cp1, tp1, tp2, cp2])

        strip_boxes += [np.array([tp0, tp1, tp2, tp3])]
        tmp = cv2.drawContours(tmp, [sbox], 0, (0, 0, 255), 5)

    if plotting:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        plt.show()

    '''
    # Processing Step 6: Extract the strips into separate images
    '''
    ref_box = np.array([[0, 0], [0, stripHeight], [stripWidth, stripHeight], [stripWidth, 0]],
                       dtype=float)

    lower_green = np.array([20, 50, 50])
    upper_green = np.array([83, 255, 255])

    if plotting:
        fig, plots = plt.subplots(1, len(strip_boxes) * 3, figsize=(10, 10))
        plt.show()

    idx = 0
    tmp = image.copy()
    raw_strip_images = []
    for sbx in strip_boxes:
        center = (statistics.mean([sbx[0][0], sbx[1][0], sbx[2][0], sbx[3][0]]),
                  statistics.mean([sbx[0][1], sbx[1][1], sbx[2][1], sbx[3][1]]))
        angle = -1 * np.degrees(np.arctan2(sbx[0][0] - sbx[1][0], sbx[0][1] - sbx[1][1]))
        # angle = angle if sbx[0][0] < sbx[1][0] else angle*-1
        width = int(pointDistance(sbx[1], sbx[2]))
        height = int(pointDistance(sbx[0], sbx[1]))

        straigtened_strip = rotate_image(image, center, angle, width,
                                         height)
        # Spurious hits often occur at the edges of the strip, so let's
        # crop them away
        cropped_strip = straigtened_strip[:,
                        int(straigtened_strip.shape[1] * 0.1):
                        int(straigtened_strip.shape[1] * 0.9)]

        if plotting:
            plots[idx].imshow(cv2.cvtColor(cropped_strip, cv2.COLOR_BGR2RGB))

        hsv = cv2.cvtColor(cropped_strip, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        cnts = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # Create a copy of the image to draw the bounding boxes on
        tmp = cropped_strip.copy()
        colored_box_cutoff = tmp.shape[0]
        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] == 0: continue

            rect = cv2.minAreaRect(c)
            box = makeOrderedBox(rect)
            if boxArea(box) > 0.10 * greenBoxArea:
                if box[0][1] < colored_box_cutoff:
                    colored_box_cutoff = box[0][1]
            tmp = cv2.drawContours(tmp, [box], 0, (0, 0, 255), 10)
            tmp = cv2.circle(tmp, (box[0][0], box[0][1]), 10, (255, 0, 0), -1)

        if plotting:
            plots[idx + 0].imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))

        colored_box_cutoff = colored_box_cutoff
        final_strip = straigtened_strip[0:colored_box_cutoff, :]

        if plotting:
            plots[idx + 1].imshow(cv2.cvtColor(final_strip, cv2.COLOR_BGR2RGB))
        #     h, status = cv2.findHomography(final_strip, ref_box)
        #     img = cv2.warpPerspective(image, h, (stripWidth, stripHeight))
        #     raw_strip_images += [img]
        height = final_strip.shape[0]
        width = final_strip.shape[1]
        h, status = cv2.findHomography(np.array([(0, height), (0, 0), (width, 0), (width, height)]),
                                       ref_box)
        img = cv2.warpPerspective(final_strip, h, (stripWidth, stripHeight))
        raw_strip_images += [img]

        if plotting:
            plots[idx + 2].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        idx += 3

    '''
    # Processing Step 7: Crop the images to remove the grip of the strips, and the vertical borders
    '''
    norm_strip_images = []

    if plotting:
        fig, plots = plt.subplots(1, len(raw_strip_images), figsize=(10, 10))
        plt.show()
    idx = 0
    tick_labels = [""]
    tick_labels.extend([str(i * 100) for i in range(stripHoldY // 100, 0, -1)])
    for img in raw_strip_images:
        # Crop out the top and the bottom parts of the strip, and applying bilateral filtering for smoothing
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
        x0 = int(marginFraction * stripWidth)
        x1 = int((1 - marginFraction) * stripWidth)
        y0 = 0
        y1 = stripHoldY

        crop = img[y0:y1, x0:x1]
        nimg = cv2.bilateralFilter(crop, 9, 75, 75)
        nimg = nimg[30:, :]
        norm_strip_images += [nimg]

        vimg = cv2.flip(nimg, 0)

        if plotting:
            plots[idx].imshow(cv2.cvtColor(vimg, cv2.COLOR_BGR2GRAY), cmap='gray')
            plots[idx].set_yticklabels(tick_labels)
            plots[idx].set_xticklabels([])
        idx += 1

    # Calculate threshold from corrected control image
    idx = len(norm_strip_images) - 1
    min0 = 0
    img0 = correct_input_image(norm_strip_images[len(norm_strip_images) - 1], 'clahe')
    data0 = img0.astype('int32')
    min0, _, min_pos = getmin(data0)
    x = np.linspace(0, 10, 50)
    y = 5 * x + 10 + (np.random.random(len(x)) - 0.5) * 5
    baseline = np.mean(data0[400: 550])
    threshold = baseline - min0

    scores = []
    for i in range(0, len(norm_strip_images) - 1):
        nimg = correct_input_image(norm_strip_images[i], 'clahe')
        scores.append(predict(nimg.astype('int32'), threshold))

    return scores, strip_boxes


def main():
    parser = argparse.ArgumentParser('Read Sherlock Strips')
    parser.add_argument('--image_file', required=True)
    parser.add_argument('--strip_pixels', type=float)
    parser.add_argument('--plotting', help="Enable plotting", action='store_true')
    args = parser.parse_args()

    scores, strip_boxes = getPredictions(args.image_file, args.strip_pixels, args.plotting)
    class_threshold = 0.7
    truths = ['POSITIVE' if s > class_threshold else 'NEGATIVE' for s in scores]
    truths.append('CONTROL')

    # The results string has the following format:
    # [{(x1,y1):(x2,y2):(x3,y3):(x4,y4);(x1,y1):(x2,y2):(x3,y3):(x4,y4)
    #      <results>POSITIVE,CONTROL]
    # where the coordnates are bottom left, top left, top right, and bottom right
    # coordinates of each strip.

    results_string = '[{'
    for strip in strip_boxes:
        for corner in strip:
            results_string += "(" + str(corner[0]) + "," + str(corner[1]) + "):"
        results_string = results_string[:-1]  # get rid of trailing colon
        results_string += ";"
    results_string = results_string[:-1]  # get rid of trailing semicolon
    results_string += "}<results>"
    for result in truths:
        results_string += result + ","
    results_string = results_string[:-1]  # get rid of trailing comma
    results_string += ']'
    print(results_string)


if __name__ == '__main__':
    main()
