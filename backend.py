
import os
from ultralytics import RTDETR , YOLO
import numpy as np
import cv2
import shutil

model = RTDETR("model_detection.pt")
model_cls = YOLO('model_classify.pt')


def smallest_box(boxes):
    if not boxes:
        return None  # Return None if the list is empty

    smallest = None
    smallest_area = float('inf')  # Initialize to infinity

    for box in boxes:
        x_center, y_center, width, height = box
        
        # Calculate the area of the box
        area = width * height
        
        # Check if this box is smaller than the current smallest
        if area < smallest_area:
            smallest_area = area
            smallest = box

    return smallest


def mean_center_largest_box(boxes):

    if not boxes:
        return None  # Return None if the list is empty

    # Initialize variables to calculate mean and largest dimensions
    total_x_center = 0
    total_y_center = 0
    largest_width = 0
    largest_height = 0

    # Iterate through the boxes to calculate mean x_center, y_center and largest width, height
    for box in boxes:
        x_center, y_center, width, height = box
        total_x_center += x_center
        total_y_center += y_center
        if width > largest_width:
            largest_width = width
        if height > largest_height:
            largest_height = height

    # Calculate mean x_center and y_center
    mean_x_center = total_x_center / len(boxes)
    mean_y_center = total_y_center / len(boxes)

    # Return the new box
    return [mean_x_center, mean_y_center, largest_width, largest_height]

import cv2
import numpy as np

def crop_and_draw_boxes(image_paths, box):

    original_images_with_box = []
    cropped_images = []

    # Unpack the box dimensions
    x_center, y_center, box_width, box_height = box

    for path in image_paths:
        # Load the grayscale image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # Skip if the image couldn't be loaded

        height, width = img.shape

        # Convert normalized coordinates to pixel values
        x_min = int((x_center - box_width / 2) * width)
        y_min = int((y_center - box_height / 2) * height)
        x_max = int((x_center + box_width / 2) * width)
        y_max = int((y_center + box_height / 2) * height)

        # Ensure the coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)

        # Crop the image
        cropped_img = img[y_min:y_max, x_min:x_max]

        # Convert grayscale image to BGR for drawing colored rectangle
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw the rectangle (bounding box) on the original image
        cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box with thickness of 2

        # Append the images to the respective lists
        original_images_with_box.append(img_bgr)
        cropped_images.append(cropped_img)

    return original_images_with_box, cropped_images


def find_min_intensity_images(levels):

    min_intensity_images = {}

    for level, images in levels.items():
        if not images:  # If the list is empty, skip to the next level
            continue
        # Find the image with the minimum intensity
        min_image = min(images, key=lambda x: x[1])
        min_intensity_images[level] = min_image[0]

    return min_intensity_images


import cv2


def get_perfect_slices(input_path):
    names = dict(model.names)

    # Assuming p[0].boxes.cls.tolist() is a list of class indices


    # Initialize slices dictionary
    slices = {
        'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': []
    }

    boxes_d = {
        'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': []
    }

    predicted_images = {}
    import matplotlib.pyplot as plt 

    for i in os.listdir(input_path):
        im = os.path.join(input_path , i)
        p = model.predict(im)
        predicted_images[im] = p[0].plot()

        classes = [names[i] for i in p[0].boxes.cls.tolist()]
        boxess = p[0].boxes.xywhn.tolist()
        box = smallest_box(boxess)
        box[1] = box[1] - 0.08
        if 'C2-C3' in classes:
            slices['C2'].append(im)
            boxes_d['C2'].append(box)
        if 'C3-C4' in classes:
            slices['C3'].append(im)
            boxes_d['C3'].append(box)
        if 'C4-C5' in classes:
            slices['C4'].append(im)
            boxes_d['C4'].append(box)
        if 'C5-C6' in classes:
            slices['C5'].append(im)
            boxes_d['C5'].append(box)
        if 'C6-C7' in classes:
            slices['C6'].append(im)
            boxes_d['C6'].append(box)
        if 'C7-T1' in classes:
            slices['C7'].append(im)
            boxes_d['C7'].append(box)

    box_lvl = {}

    slices
    for lvl in slices:
        if len(boxes_d[lvl]) > 0:
            boxes = boxes_d[lvl]
            box = mean_center_largest_box(boxes)
            box_lvl[lvl] = box

    crr_images = {}
    org_images = {}

    for lvl in box_lvl:
        img_paths = slices[lvl]
        
        if(box_lvl[lvl]) != None:
            original_images_with_box, cropped_images = crop_and_draw_boxes(img_paths , box_lvl[lvl])
            crr_images[lvl] = cropped_images
            org_images[lvl] = original_images_with_box


    images_int_based_on_level = {
        'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': []
    }

    for lvl in crr_images:
        for i , name in zip(crr_images[lvl] , slices[lvl]) : 
            images_int_based_on_level[lvl].append((name , (np.array(i).sum())))


    disks = find_min_intensity_images(images_int_based_on_level)
    return disks


def crop_bboxes(results):
    img = results[0].orig_img  # Original image
    classes = results[0].boxes.cpu().cls.numpy()  # Predicted classes
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)  # Bounding boxes

    cropped_images = []

    for cls, bbox in zip(classes, boxes):
        if cls in [0, 1]:
            # Crop the region of interest (ROI) corresponding to the bounding box
            x_min, y_min, x_max, y_max = bbox
            ROI_box = bbox
            cropped_img = img[y_min:y_max, x_min:x_max]

            if cropped_img.size == 0:
                continue  # Skip invalid crops

            cropped_images.append(cropped_img)

    if not cropped_images:
        raise Exception('No bounding boxes for classes 0 or 1 were detected.')

    return cropped_images , ROI_box


def plot_bboxes(results):
    img = results[0].orig_img # original image
    names = results[0].names # class names dict
    scores = results[0].boxes.cpu().conf.numpy() # probabilities
    classes = results[0].boxes.cpu().cls.numpy() # predicted classes
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32) # bboxes
    lbl_margin = 3 

    cropped_images , ROI_bbox = crop_bboxes(results)

    ROI  = cropped_images[0]
    
    classify_results = model_cls.predict(ROI)
    
    diag = None
    
    if classify_results[0].probs.top1 == 0:
        label = 'CDH'
        diag = label
        color = (0, 0, 255)
       
    else:
        label = 'Normal'
        diag = label
        color = (0, 255, 0) 
        
       
    height, width = img.shape[:2] 

    img = cv2.rectangle(img, (ROI_bbox[0], ROI_bbox[1]),
                        (ROI_bbox[2], ROI_bbox[3]),
                        color=color,
                        thickness=1)
        
    label_size = cv2.getTextSize(label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1)
    lbl_w, lbl_h = label_size[0] 
    lbl_w += 2 * lbl_margin 
    lbl_h += 2 * lbl_margin
    
    
    img = cv2.rectangle(img, (ROI_bbox[0], ROI_bbox[1]), 
                         (ROI_bbox[0] + lbl_w, ROI_bbox[1] - lbl_h),
                         color=color, 
                         thickness=-1)

    cv2.putText(img, label, (ROI_bbox[0] + lbl_margin, ROI_bbox[1] - lbl_margin),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0, color=(255, 255, 255),
                thickness=1)
    

    
    levels = [2,3,4,5,6,7]
    
    for score, cls, bbox in zip(scores, classes, boxes): 
        class_label = names[cls]
        if  cls in levels:
            lvl = class_label
   
    if diag==None:
        shutil.rmtree('uploads')
        raise Exception('No Cervical Axial MRI was detected; check entered images')
    
    return img , lvl , diag 




def generate_report(patient_id, age, gender, levels, diag):
    report = f"Diagnosis Report:\n"
    
    # List to store levels with CDH
    cdh_levels = []
    
    # Iterate through the levels and diagnoses
    for level, diagnosis in zip(levels, diag):
        report += f"Level {level}: {diagnosis}\n"
        
        # Check if CDH is detected
        if 'CDH' in diagnosis.upper():
            cdh_levels.append(level)  # Add level to the list of CDH levels
    
    # If there are any levels with CDH, add the final statement with the recommendation
    if cdh_levels:
        report += f"\nThere appears to be CDH for the following levels: {', '.join(cdh_levels)}. \n"
    
    # Add a note at the end that the report was generated by Spine AI
    report += "\n\nThis report was generated by Spinal AI, an advanced AI tool for spine diagnosis."
    
    return report


