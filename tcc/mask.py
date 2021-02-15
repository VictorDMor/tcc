import cv2
import numpy as np

def check_threshold(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_lower = np.array([30, 40, 40])
    green_higher = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, green_lower, green_higher)
    green_pixels_amount = image[mask > 0].shape[0]
    print(green_pixels_amount/(image.shape[0] * image.shape[1]))
    image[mask > 0] = (0, 0, 255)
    cv2.imshow('Image', image)
    cv2.waitKey(0)


check_threshold('mask_teste_1.png')