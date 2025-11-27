import cv2
import numpy as np

def pre_process_img(number_roi):
    resized_img_char = cv2.resize(number_roi, (64,64), interpolation=cv2.INTER_AREA)
    gray_scaled_img=cv2.cvtColor(resized_img_char,cv2.COLOR_BGR2GRAY)
    ret, binary_image_otsu = cv2.threshold(
    gray_scaled_img, 
    0,                               # The threshold is ignored when using THRESH_OTSU
    255,                             # Maximum value to assign (white)
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)   
    return binary_image_otsu