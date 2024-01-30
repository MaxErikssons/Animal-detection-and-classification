import os
import cv2
from matplotlib import pyplot as plt

# Define the directory paths
image_dir = 'D:/Exjobb/Kod/train_data/images/train'
label_dir = 'D:/Exjobb/Kod/train_data/labels/train'
save_dir = 'D:/Exjobb/Kod/train_data/cropped'  # Directory to save cropped images and labels

# Ensure the save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Function to get bounding box coordinates and class label from a label file
def get_bboxes_and_label(label_path):
    with open(label_path, 'r') as file:
        data = file.readlines()
    labels = [int(x.strip().split()[0]) for x in data]  # Extract class labels
    bboxes = [list(map(float, x.strip().split()))[1:] for x in data]  # Convert to float and skip the class
    return bboxes, labels

# Function to convert normalized bbox coordinates to pixel coordinates
def unnormalize_bbox(bbox, img_width, img_height):
    x_center, y_center, width, height = bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    return x1, y1, x2, y2

# Function to crop image based on the bounding box
def crop_image(img_path, bbox):
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]
    x1, y1, x2, y2 = unnormalize_bbox(bbox, img_width, img_height)
    crop_img = img[y1:y2, x1:x2]
    return crop_img

# Script to read images, crop based on bbox, save cropped image and label
for i, filename in enumerate(os.listdir(image_dir)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check for image files
        # Construct paths to image and label
        img_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')
        
        # Check if the corresponding label file exists
        if os.path.exists(label_path):
            # Get bounding boxes and labels
            bboxes, labels = get_bboxes_and_label(label_path)
            # Crop and save each bbox in the image along with its label
            for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
                cropped_img = crop_image(img_path, bbox)
                cropped_img_filename = f"{os.path.splitext(filename)[0]}_cropped_{idx}.jpg"
                cropped_img_path = os.path.join(save_dir, cropped_img_filename)
                cv2.imwrite(cropped_img_path, cropped_img)  # Save cropped image
                
                # Save the label file for the cropped image
                label_filename = f"{os.path.splitext(cropped_img_filename)[0]}.txt"
                label_path = os.path.join(save_dir, label_filename)
                with open(label_path, 'w') as label_file:
                    label_file.write(str(label))