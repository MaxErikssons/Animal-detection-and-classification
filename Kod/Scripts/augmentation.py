import os
import cv2
import yaml
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load constants from YAML file
with open("constants.yaml", 'r') as stream:
    CONSTANTS = yaml.safe_load(stream)

def is_image_by_extension(file_name):
    image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']
    return file_name.lower().split('.')[-1] in image_extensions

def load_image_and_labels(img_file):
    base_name = os.path.splitext(img_file)[0]
    aug_file_name = f"{base_name}_{CONSTANTS['transformed_file_name']}"
    image = cv2.imread(os.path.join(CONSTANTS["inp_img_pth"], img_file))
    label_path = os.path.join(CONSTANTS["inp_lab_pth"], f"{base_name}.txt")
    gt_bboxes = load_labels(label_path)
    return image, gt_bboxes, aug_file_name

def load_labels(label_path):
    with open(label_path, "r") as file:
        lines = [line.strip() for line in file if line.strip()]
        return [parse_label(line) for line in lines] if lines else []

def parse_label(label):
    parts = label.split()
    class_id = int(parts[0])
    class_name = CONSTANTS['CLASSES'][class_id]
    bbox = list(map(float, parts[1:]))
    return bbox + [class_name]

def convert_to_yolo_format(bboxes):
    return [convert_bbox(bbox) for bbox in bboxes]

def convert_bbox(bbox):
    class_name = bbox[-1]
    class_id = CONSTANTS['CLASSES'].index(class_name)
    return [class_id] + list(bbox[:-1])  # Convert tuple to list before concatenation

def save_label(bboxes, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    path = os.path.join(output_dir, f"{file_name}.txt")
    with open(path, 'w') as file:
        for bbox in bboxes:
            bbox_str = ' '.join(map(str, bbox))
            file.write(f"{bbox_str}\n")

def save_image(image, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    path = os.path.join(output_dir, f"{file_name}.png")
    cv2.imwrite(path, image)

def apply_augmentations(image, bboxes):
    # https://albumentations.ai/docs/api_reference/full_reference/
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0),
        A.CLAHE(clip_limit=(0, 1), tile_grid_size=(8, 8), always_apply=True),
        A.Resize(300, 300)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    class_labels = [bbox[-1] for bbox in bboxes]
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    return transformed['image'], transformed['bboxes']

def save_augmentation(image, bboxes, file_name):
    if bboxes:
        yolo_bboxes = convert_to_yolo_format(bboxes)
        save_label(yolo_bboxes, CONSTANTS["out_lab_pth"], file_name)
        save_image(image, CONSTANTS["out_img_pth"], file_name)
    else:
        print("No objects to save.")

def plot_image_with_bboxes(image, bboxes, title='Image with Bounding Boxes'):
    plt.figure()  # Create a new figure
    ax = plt.gca()  # Get the current Axes instance on the current figure
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for bbox in bboxes:
        x_center, y_center, width, height, _ = bbox
        rect = patches.Rectangle(
            ((x_center - width / 2) * image.shape[1], (y_center - height / 2) * image.shape[0]),
            width * image.shape[1],
            height * image.shape[0],
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

    plt.title(title)
    plt.show(block=False)  # Display the figure without blocking

def run_augmentation():
    images = [img for img in os.listdir(CONSTANTS["inp_img_pth"]) if is_image_by_extension(img)]
    for img_num, img_file in enumerate(images):
        image, bboxes, file_name = load_image_and_labels(img_file)
        plot_image_with_bboxes(image, bboxes, title=f'Original Image {img_num + 1}')  # Plot original image with bounding boxes
        augmented_image, augmented_bboxes = apply_augmentations(image, bboxes)
        save_augmentation(augmented_image, augmented_bboxes, file_name)
        plot_image_with_bboxes(augmented_image, augmented_bboxes, title=f'Augmented Image {img_num + 1}')  # Plot augmented image with bounding boxes

    plt.show()  # Keep all figures open until closed manually

if __name__ == "__main__":
    run_augmentation()
