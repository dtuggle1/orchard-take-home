import pandas as pd
from torch.backends.cudnn import deterministic

[ ]
import os
from download import *
import cv2
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from helpers import *
import copy
from PIL import Image
from ultralytics import YOLO
import torch
import pickle


from ultralytics.utils.plotting import Colors
colors = Colors()
for i,color in enumerate(Colors().palette):
    color = (0,0,255)
print()
# Code taken from AI to change ultralytics to plot only red
# from ultralytics.utils.plotting import colors
#
# # Monkey patch the colors function to always return red
# def red_only_colors(i, bgr=False):
#     """Return red color for any class index"""
#     return (0, 0, 255) if bgr else (255, 0, 0)

# Temporarily override the colors function
# original_colors = colors
# import ultralytics.utils.plotting
# ultralytics.utils.plotting.colors = red_only_colors

evaluation_set_rgb_folder_link = "https://drive.google.com/drive/folders/1Ua9R3pC5HZdiUKPGdoCpS_MKmy6O4_-p?usp=drive_link"


# This downloads all of the Images in the EvaluationSetRGB google drive folder to Colab's local filesystem. In this case,
# 49 files should be downloaded from the RGB folder on google drive, to "/content/EvaluationSetRGB/" locally, stored as Images 001 to 049

# Check that an image exists in the Colab filesystem

# Add Evaluation Set Files
EVALUATION_SET_RGB_FOLDER_LINK = "https://drive.google.com/drive/folders/1Ua9R3pC5HZdiUKPGdoCpS_MKmy6O4_-p?usp=drive_link"
EVALUATION_SET_DEPTH_FOLDER_LINK = "https://drive.google.com/drive/folders/1V_rZAPt13EFp1k4ouwsSLj9_Fh9L_red?usp=drive_link"
EVALUATION_SET_RGB_FOLDER_NAME = 'EvaluationSetRGB'
EVALUATION_SET_DEPTH_FOLDER_NAME = 'EvaluationSetDepth'
EVALUATION_SET_OUTPUTS_FOLDER_NAME = 'EvaluationSetOutputs'
P4_OUTPUTS_FOLDER_NAME = 'P4Outputs'
P4_LABELS_FOLDER_NAME = 'P4Labels'
TRAINING_DATA_FOLDER_NAME = 'Training Data'
MASKED_TRAINING_DATA_FOLDER_NAME = 'MaskedTrainingData'
CONF_THRESH = .35

SPLIT_TRAINING_DATA = True
EVALUATE_EVALUATION_SET_P2 = True
OVERLAY_LABELS_ON_TRAIN_DATA = True

BEST_MODEL = os.path.join('runs','segment','train35','weights','best.pt')
TRAIN_MODEL = False




def part1():
    IMAGE_INDEX = 25
    rgb_filenames = {f"0{str(i).zfill(2)}_rgb.jpg" for i in
                     range(1, 50)}  # set containing all of the rgb folder filenames
    depth_filenames = {f"0{str(i).zfill(2)}_depth.png" for i in
                       range(1, 50)}  # set containing all of the depth folder filenames

    download_image_folder(EVALUATION_SET_RGB_FOLDER_NAME,
                          EVALUATION_SET_RGB_FOLDER_LINK, rgb_filenames)
    download_image_folder(EVALUATION_SET_DEPTH_FOLDER_NAME,
                          EVALUATION_SET_DEPTH_FOLDER_LINK, depth_filenames)

    input_image_filename = f"{str(IMAGE_INDEX).zfill(3)}_rgb.jpg"
    input_image = cv2.imread(os.path.join(EVALUATION_SET_RGB_FOLDER_NAME, input_image_filename))


    # The RGB color of the 100th pixel from the left, 200th pixel from the top in the 25th image chronologically.
    X_INDEX = 100
    Y_INDEX = 200

    rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    rgb_color = rgb_image[Y_INDEX][X_INDEX]
    print(f'The RGB color of the 100th pixel from the left, 200th pixel from the top in the 25th image chronologically: ({rgb_color})')

    # The depth of this same pixel, via the depth map, returned in millimeters.
    depth_filename = get_depth_filename_from_image_filename(input_image_filename)
    depth_image = cv2.imread(os.path.join(EVALUATION_SET_DEPTH_FOLDER_NAME, depth_filename))
    depth_at_pixel = depth_image[Y_INDEX][X_INDEX][0] # all of the channels are the same so just take the first value for the depth
    depth_map = depth_image[:,:,0]
    print(f"The depth of this same pixel, via the depth map, returned in millimeters: {depth_at_pixel}")

    # The total number of pixels in this image that have depth values +/- 10 mm of the depth value you found.
    BOUND_PLUS_MINUS = 10 # bounding in mm
    mask = ((depth_map >= depth_at_pixel-BOUND_PLUS_MINUS) & (depth_map <=
                                                              depth_at_pixel+BOUND_PLUS_MINUS)).astype(int)
    print(f"The total number of pixels in this image that have depth values +/- 10 mm of the depth value you found: {np.sum(mask)}")

def part2():
    """Part 2

    In this part, you will train a machine learning model to detect tree trunks. As part of this challenge you should have received a separate folder with two sub-folders, "Images" and "Labels".

    In these folders, there are 30 Images of tree trunks, and 30 corresponding annotation files containing masks of the tree trunks.

    The amount of labelled training data will be relatively sparse, and while the trunks in the training data will look similar to the ones in the challenge's Evaluation Set, they will not be exactly the same trunks.

    Your objective is to architect, tune hyperparameters for, and train a machine learning model to predict per-pixel masks on trunks. If you wish, you may implement existing architectures such as YOLO, Mask-RCNN, etc, as a starting point, but the best results will only be achieved with fine-tuning, re-architecting, and tuning both the structure and hyperparameters of these models. You will be graded directly on the accuracy and precision of your mask detections, both on the evaluation set and on a separate, unseen test set.

    You should save your trained model and weights as part of this Colab notebook.

    In your comments, please write detailed notes on exactly how you choose to prototype and structure your model, and exactly what design decisions you made and why.

    Also, please ensure that you set appropriate random seeds so that your trained model is reproducible. We will be re-running the training code to reproduce your model upon completion of the challenge.

    Example tree trunk masks

    ExampleTrunkPredictions.png

    Below, we demonstrate how to download and load in the trunk Images and masks, as well as how to parse the mask label files to plot the masks on top of Images.

    """
    """
    # This downloads all of the Images in the EvaluationSetRGB google drive folder to Colab's local filesystem. In this case, these files will have downloaded
    # 30 Images + 30 Labels = 60 total files should be downloaded from the RGB folder on google drive, to "/content/Images/" and "/content/Labels/" locally

    # Check that an image and corresponding label exists in the Colab filesystem
    print("File downloaded: ", os.path.exists("/content/Images/130_rgb.jpg"))
    print("File downloaded: ", os.path.exists("/content/Labels/130_label.txt"))

    # The Labels are stored in the YOLOv5 contour mask format, where each line in the text file represents a separate mask label. Within each line, the first token (an integer) represents a class
    # In this case, since there is just one class "Trunk", all of the Labels have class = "0". The following N * 2 tokens (all floats) represent the coordinates of each of the points of the mask.
    # Together, the points for each label represent individual vertices of a contour / polygon that represents the mask. Each of the points/vertices is represented as "x y ", delineated with spaces.
    # In this coordinate frame, (x, y) = (0, 0) at the top left corner of the image. Each X and Y value is a float between 0 and 1, representing the proportion of the image's width/height where the vertex occurs.
    # i.e. (x, y) = (0.25, 0.5) would be the point 1/4 from the left of the image, and halfway down the image from the top.

    # Each label textfile may have 0, 1, 2, or more masks, each on a new line of the text file.
    # Below, we demonstrate how to parse the label file and plot a mask on top of an image.
    """


    ### MY CODE BELOW

    # Below is where you write your code to train your ML model to predict masks on trunks

    if OVERLAY_LABELS_ON_TRAIN_DATA:
        if MASKED_TRAINING_DATA_FOLDER_NAME not in os.listdir():
            os.mkdir(os.path.join(MASKED_TRAINING_DATA_FOLDER_NAME))

        label_folder_path = os.path.join('Training Data', 'Labels')
        training_image_folder_path = os.path.join('Training Data', 'Images')

        overlay_labels(training_image_folder_path, label_folder_path, MASKED_TRAINING_DATA_FOLDER_NAME)

    if TRAIN_MODEL:
        detector = TrunkDetector()
        detector.train()
        model = detector.model
    else:
        model = YOLO(BEST_MODEL)

    if EVALUATE_EVALUATION_SET_P2:
        if EVALUATION_SET_OUTPUTS_FOLDER_NAME in os.listdir():
            shutil.rmtree(EVALUATION_SET_OUTPUTS_FOLDER_NAME)
        os.mkdir(EVALUATION_SET_OUTPUTS_FOLDER_NAME)

        eval_images = []
        for eval_image_filename in os.listdir(EVALUATION_SET_RGB_FOLDER_NAME):
            eval_images.append(os.path.join(EVALUATION_SET_RGB_FOLDER_NAME, eval_image_filename))

        output_images, output_results = process_images(model, eval_images, save_folder=EVALUATION_SET_OUTPUTS_FOLDER_NAME)


    # Things I would do with more time
    # Detect ground/soil and verify that the the bottom of trunks are embedded in the
    # ground.
    return output_images, output_results

def part3(model):
    """
    Part 3 In this part, you should provide a code block with inference code, where an input image can be pulled from the internet, be inferenced on, and output + display a result image with a semantically segmented mask of the detected trunk(s), which is displayed in the Colab notebook.

    For example, you can test using the example input url https://ik.imagekit.io/orchardrobotics/example_trunk_to_inference.jpg. If this url is put in the inference code block of this Colab notebook, your code should automatically download the image, use it as input to your machine learning model, and then display a resulting image representing where the trunk(s) are.

    The output image should be a pixel-wise mask with only two colors occuring in the image. Where white pixels (255, 255, 255) represent pixels that are part of any trunk mask, and black pixels (0, 0, 0) represent everything else.
    """


    # URL of the image
    url = 'https://ik.imagekit.io/orchardrobotics/example_trunk_to_inference.jpg'

    # Fetch the image
    response = requests.get(url)
    original_image = Image.open(BytesIO(response.content))


    ### Run inference using your model here ###
    # ml_inference_output should be the pixel-wise mask, with each pixel either (0, 0, 0) or (255, 255, 255) depending on whether it is part of a trunk mask
    output_images, output_results = process_images(model, original_image)

    # Save the image
    ml_inference_output = generate_combined_mask(output_results[0][0])
    print("ML Inference Output Image:")
    # Image.show(ml_inference_output)

    # Save the image
    cv2.imwrite('ml_inference_output_image.jpg',ml_inference_output)

def part4(results):
    """
    Part 4

    In this part, you will need to develop a method to count the total number of trunks in the sequence of Evaluation Set Images.

    A portion of this will involve using your trained ML model to inference on the RGB Images of the trunks in the EvaluationSetRGB folder, to detect the trunks.

    Afterwards, your method must be able to count the number of unique trunks in the sequence of Images, without undercounting or double-counting any given trunk.

    The result of this part should be one number, indicating the total number of trunks in the sequence of 49 Images.

    In addition, output the inferences of your ML model on each of the 49 Images, with the tree mask visible in translucent red, plotted on top of the original RGB image. On top of each mask, plot text indicating the unique ID of each tree. (i.e. if there are 20 unique trees in the sequence of 49 Images, the first 4 Images might all contain tree 1, the 5th image might contain tree 1 and tree 2, the next 5 Images might only contain tree 2, etc. Plot the unique ID assigned to each tree on top of that tree's mask).
    """
    # Count trunks here
    for result in results:
        if len(result) < 1:
            continue
        result.masks.sizes = []
        result.masks.centroids = []
        result.masks.ids = []

        for mask in result.masks.data:
            centroid = mask_centroid(mask)
            result.masks.centroids.append(centroid)
            result.masks.sizes.append(np.sum(mask.cpu().numpy()))
    camera_direction = group_masks(results, determine_direction=True)
    print(f'Determined Camera Direction: {camera_direction}')
    pairing_table = group_masks(results, direction=camera_direction)
    pairing_table.to_csv('pairing_table.csv')
    tree_idx = 0
    pairing_table = pd.read_csv('pairing_table.csv')
    groups = []
    for i, row in pairing_table.iterrows():
        val1 = (int(row['result idx']), int(row['result mask idx']))
        val2 = (int(row['comparison result idx']), int(row['comparison result mask idx']))
        if i == 0:
            groups.append([])
            groups[0].append(val1)
            groups[0].append(val2)
            continue
        found_group = False
        for group in groups:
            if val1 in group:
                found_group = True
                if val2 not in group:
                    group.append(val2)
                break
        if not found_group:
            groups.append([])
            groups[-1].append(val1)
            groups[-1].append(val2)

    # special logic for last image because if a trunk appears, it is impossible for it to be part of a group
    for j,mask in enumerate(results[-1].masks.data):
        mask_id = (len(results)-1,j)
        mask_id_in_a_group = False
        for group in groups:
            if mask_id in group:
                mask_id_in_a_group = True
        if not mask_id_in_a_group:
            groups.append([mask_id])


    class_labels = {}
    for i,group in enumerate(groups):
        class_labels[i] = f"Trunk_{i}"

    for i, result in enumerate(results):
        if len(result) < 1:
            continue
        keep_mask_indeces = []
        for j, mask in enumerate(result.masks.data):
            mask_id = (i,j)
            for k, group in enumerate(groups):
                if mask_id in group:
                    result.boxes.cls[j] = k
                    keep_mask_indeces.append(j)
                    continue

        result.masks = result.masks[keep_mask_indeces]
        result.boxes = result.boxes[keep_mask_indeces]

    # save the labels to a folder
    if P4_OUTPUTS_FOLDER_NAME in os.listdir():
        shutil.rmtree(P4_OUTPUTS_FOLDER_NAME)
    os.mkdir(P4_OUTPUTS_FOLDER_NAME)

    def place_label_safely(image, text, x, y, font, font_scale, thickness, bg_color=(255, 255, 255),
                           text_color=(0, 0, 0)):
        """Place a label with background that stays within image bounds"""
        img_h, img_w = image.shape[:2]

        # Get text dimensions
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        padding = 5

        # Try different positions in order of preference
        positions = [
            # Above the object
            (x, y - text_height - 10),
            # Below the object
            (x, y + 50),
            # Left side
            (max(5, x - text_width - 20), y),
            # Right side
            (x + 20, y),
            # Top-left corner as fallback
            (10, text_height + 10),
        ]

        for label_x, label_y in positions:
            # Calculate background rectangle coordinates
            bg_x1 = label_x - padding
            bg_y1 = label_y - text_height - padding
            bg_x2 = label_x + text_width + padding
            bg_y2 = label_y + baseline + padding

            # Check if this position fits within image bounds
            if (bg_x1 >= 0 and bg_y1 >= 0 and bg_x2 <= img_w and bg_y2 <= img_h):
                # Draw background rectangle
                cv2.rectangle(image, (int(bg_x1), int(bg_y1)), (int(bg_x2), int(bg_y2)), bg_color, -1)

                # Draw text (coordinates should align with background)
                cv2.putText(image, text, (int(label_x), int(label_y)), font, font_scale, text_color, thickness)
                return

        # Fallback: force it in the top-left corner with proper alignment
        label_x = 10
        label_y = text_height + 10
        bg_x1 = 5
        bg_y1 = 5
        bg_x2 = min(img_w - 5, text_width + 15)
        bg_y2 = min(img_h - 5, text_height + 15)

        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        cv2.putText(image, text, (label_x, label_y), font, font_scale, text_color, thickness)

    # Alternative simpler version that's more reliable:


    # Use the simpler version in your main code:
    for i, result in enumerate(results):
        if len(result) < 1:
            continue
        original_image = cv2.imread(result.path)
        output_image = original_image.copy()

        for j, mask in enumerate(result.masks.data):
            if hasattr(result.masks, 'xy') and len(result.masks.xy) > j:
                polygon = result.masks.xy[j]

                if len(polygon) > 0:
                    # Your existing polygon processing code...
                    polygon = polygon.astype(np.float32)

                    valid_mask = (
                            np.isfinite(polygon).all(axis=1) &
                            (polygon[:, 0] >= 0) & (polygon[:, 0] < output_image.shape[1]) &
                            (polygon[:, 1] >= 0) & (polygon[:, 1] < output_image.shape[0])
                    )
                    polygon = polygon[valid_mask]

                    if len(polygon) < 3:
                        continue

                    polygon = polygon[~np.all(np.diff(polygon, axis=0, prepend=polygon[-1:]) == 0, axis=1)]

                    if len(polygon) < 3:
                        continue

                    polygon = polygon.astype(np.int32)
                    polygon[:, 0] = np.clip(polygon[:, 0], 0, output_image.shape[1] - 1)
                    polygon[:, 1] = np.clip(polygon[:, 1], 0, output_image.shape[0] - 1)

                    if len(np.unique(polygon, axis=0)) < 3:
                        continue

                    try:
                        mask_img = np.zeros((output_image.shape[0], output_image.shape[1]), dtype=np.uint8)
                        cv2.fillPoly(mask_img, [polygon], 255)

                        if np.sum(mask_img) == 0:
                            continue

                        color = (0, 0, 255)
                        alpha = 0.4

                        # Apply overlay using original image for consistent translucency
                        mask_indices = mask_img == 255
                        output_image[mask_indices] = (
                                original_image[mask_indices] * (1 - alpha) +
                                np.array(color) * alpha
                        ).astype(np.uint8)

                        # Draw contours
                        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            cv2.drawContours(output_image, contours, -1, color, 2)

                            # Add label with the simpler function
                            x, y, w, h = cv2.boundingRect(contours[0])
                            label = class_labels[int(result.boxes.cls[j].cpu().numpy())]

                            place_label_simple(output_image, label, x, y,
                                               cv2.FONT_HERSHEY_SIMPLEX, 2, 2)

                    except Exception as e:
                        print(f"Error processing polygon for mask {j}: {e}")
                        continue

        cv2.imwrite(os.path.join(P4_OUTPUTS_FOLDER_NAME, f"{result.path.split('\\')[1][:3]}_p4output.jpg"),
                    output_image)

    # Print total number of trunks

    # Display each of the 49 annotated Images with masks in translucent red on top of the RGB image, and unique tree IDs on top of each detected mask

def part5():
    """
Part 5 (Bonus Points!)

In this part, you will need to infer the locations of each tree trunk.

It will be important to utilize both your mask predictions and the information from each image's depth map (stored in EvaluationSetDepth).

To accomplish this, you will need a way to localize each unique trunk you see, and then convert those coordinates into a global coordinate frame.

Assume that the location of the first tree is (x, y) = (0 meters, 0 meters), and the camera is moving directly from east to west (i.e. to the left, in the direction of negative X).
"""

    # Infer locations of trunks here
    # Plot the tree trunk locations below using matplotlab, using the starter code. The
    # first tree trunk should be at (0,0). The gridlines are automatically set to be at 1 meter intervals.


    # Defining a couple example points (replace these with your predicted tree trunk locations)
    points = [(0, 0), (-5, 0), (-10, 5)]

    # Plotting the points
    for x, y in points:
        plt.scatter(x, y, c='red')
        plt.text(x, y, f'({x}, {y})', fontsize=12)

    # Setting gridlines every 1 meter
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(np.arange(min(-10, -10), max(5, 5) + 1, 1))
    plt.yticks(np.arange(min(0, -5), max(5, 5) + 1, 1))

    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.show()

if __name__ == '__main__':
    # part1()
    # if SPLIT_TRAINING_DATA:
    #     split_training_data(0.8)
    # output_images, output_results = part2()
    # with open("p2_output_results.pkl", "wb") as f:
    #     pickle.dump(output_results, f)
    # part3(YOLO(BEST_MODEL))
    with open("p2_output_results.pkl", "rb") as f:
        output_results = pickle.load(f)
    part4(output_results)
    # part5()