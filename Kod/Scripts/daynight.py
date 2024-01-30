import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

# KÄLLA:
# https://stackoverflow.com/questions/77261850/how-can-i-determine-if-a-frame-image-in-a-video-is-in-night-vision-mode-using-co
# Om RGB-kanalerna har samma värde (grå) är det troligtvis en nattbild! 

currentFolder = "NINAVarg"

def classify_day_night(image_path):
    image = cv2.imread(image_path) if isinstance(image_path, str) else image_path
    if(image is not None):
        h, w, c = image.shape

        # Coordinates for the center pixel and its four surrounding pixels (distance 10)
        center = (h // 2, w // 2)
        up = (h // 2 - 10, w // 2)
        down = (h // 2 + 10, w // 2)
        left = (h // 2, w // 2 - 10)
        right = (h // 2, w // 2 + 10)

        # Check if all five pixels are gray (RGB values are equal)
        for coords in [center, up, down, left, right]:
            r, g, b = image[coords]
            if r != g or r != b or g != b:
                return 'day'

        return 'night'
    
    return 'fail'

    # hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    # value_sum = np.sum(hsv_image[:,:,2])
    # dimensions = h * w
    # avg_brightness = value_sum / dimensions
    # threshold = 120
    # # print(avg_brightness)
    # if avg_brightness>threshold:
    #     # Day
    #     return 'day'
    # else:
    #    # Night
    #     return 'night'

# Function to process a video and save frames
# def process_video(video_path, filename, day_folder, night_folder):
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    classification = None
    if not video_capture.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if(frame_count == 0):
            classification = classify_day_night(frame)
        frame_path = os.path.join(night_folder if classification == 'night' else day_folder, f"{filename}_frame_{frame_count:05d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    video_capture.release()
    print(f" Processed {frame_count} frames from video: {os.path.basename(video_path)}")
    os.remove(video_path)  # Delete the video file after processing all frames
    print(f"Processed and removed video: {video_path}")

# Path to the directory containing images
directory_path = "Bilder//"+currentFolder

# Create a folder to store night and day images respectievly if it doesn't exist
night_images_folder = "Bilder/"+currentFolder+"/nightimages"
os.makedirs(night_images_folder, exist_ok=True)

day_images_folder = "Bilder/"+currentFolder+"/dayimages"
os.makedirs(day_images_folder, exist_ok=True)

# Total number of images
total_images = len([name for name in os.listdir(directory_path) if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))])
processed_images = 0

# Loop over all the images in the directory
for filename in os.listdir(directory_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
        image_path = os.path.join(directory_path, filename)
        classification = classify_day_night(image_path)
        
        if classification == 'night':
            # Move the night image to the night_images_folder
            destination_path = os.path.join(night_images_folder, filename)
            os.rename(image_path, destination_path)
        elif classification == 'day':
            # Move the day image to the day_images_folder
            destination_path = os.path.join(day_images_folder, filename)
            os.rename(image_path, destination_path)
        else:
            print(image_path + ' failed')

        processed_images += 1
        progress = processed_images / total_images * 100
        sys.stdout.write(f"\rProcessed {processed_images}/{total_images} images ({progress:.2f}%)")
        sys.stdout.flush()


# # Get all video files
# video_files = [name for name in os.listdir(directory_path) if name.lower().endswith('.mp4')]
# total_videos = len(video_files)
# processed_videos = 0

# # Process each video
# for filename in video_files:
#     video_path = os.path.join(directory_path, filename)
#     process_video(video_path, filename, day_images_folder, night_images_folder)
    
#     processed_videos += 1
#     progress = processed_videos / total_videos * 100
#     sys.stdout.write(f"\rProcessed {processed_videos}/{total_videos} videos ({progress:.2f}%)")
#     sys.stdout.flush()


# print()
