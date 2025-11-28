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

def sort_boxes_spatially(boxes, y_threshold=15):
    """
    Sorts boxes by rows (Y) and then columns (X).
    y_threshold: The pixel variance allowed for items to be considered on the same 'line'.
    """
    # 1. Convert tensor/results to a standard Python list of [x1, y1, x2, y2]
    # We assume 'boxes' is the box object from YOLO
    box_list = boxes.xyxy.tolist()

    # If list is empty, return empty
    if not box_list:
        return []

    # 2. Initial Sort by Y1 (top coordinate)
    # This puts them roughly in order from top to bottom
    box_list.sort(key=lambda k: k[1])

    sorted_result = []
    current_row = [box_list[0]]

    # 3. Group by Rows
    for box in box_list[1:]:
        # Compare current box's Y1 with the last box in the current row
        # If the difference is small (within threshold), they are on the same line
        if abs(box[1] - current_row[-1][1]) < y_threshold:
            current_row.append(box)
        else:
            # The line is finished. Sort this row by X1 (Left to Right)
            current_row.sort(key=lambda k: k[0])
            sorted_result.extend(current_row)
            
            # Start a new row
            current_row = [box]

    # 4. Append the final row (don't forget this step!)
    current_row.sort(key=lambda k: k[0])
    sorted_result.extend(current_row)

    return sorted_result
