import cv2
import numpy as np


def get_area(image):
    original_img = np.array(image.convert('RGB'))
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
    # Define the lower and upper bounds for the red color range
    lower_red_1 = np.array([0, 80, 15])
    upper_red_1 = np.array([12, 255, 255])
    lower_red_2 = np.array([150, 80, 15])
    upper_red_2 = np.array([180, 255, 255])
    # Create a mask for the red color range
    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    res_mask = cv2.bitwise_or(mask_1, mask_2)
    # Count number of red pixels
    k_pixels = res_mask[res_mask > 0].shape[0]
    return k_pixels


def get_contours(image):
    original_img = np.array(image.convert('RGB'))
    # Create a mask for the red color range
    hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
    lower_red_1 = np.array([0, 80, 15])
    upper_red_1 = np.array([12, 255, 255])
    lower_red_2 = np.array([150, 80, 15])
    upper_red_2 = np.array([180, 255, 255])
    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    res_mask = cv2.bitwise_or(mask_1, mask_2)
    # Perform morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    processed_mask = res_mask.copy()
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)
    # Apply Gaussian blur
    blurred_mask = cv2.GaussianBlur(processed_mask, (5,5), 0)
    # Find contours
    contours, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours based on area
    min_contour_area = 300
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            filtered_contours.append(contour)
    # Draw contours on the image
    cv2.drawContours(original_img, filtered_contours, -1, (0, 255, 0), 2)
    return original_img


def get_artefacts(image):
    original_img = np.array(image.convert('RGB'))
    mean_shift = cv2.pyrMeanShiftFiltering(original_img, 30, 60, 3)
    # gray_mean_shift = cv2.cvtColor(mean_shift, cv2.COLOR_BGR2GRAY)
    # _, bin_mean_shift = cv2.threshold(gray_mean_shift, 60, 255, cv2.THRESH_BINARY)
    return mean_shift # bin_mean_shift
