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

def group_masks(results, direction=None, determine_direction=False):
    """
    if direction is True, output the final outputs
    if determine_direction is True, run a rough matching process (without the direction), and the median direction of the True labels will determine the direction of the cameras
    """
    if direction and determine_direction:
        raise ValueError('Cannot both determine the direction and use direction for final outputs')
    if not direction and not determine_direction:
        raise ValueError('Must be ')
    frames_comparison = 10
    true_x_distances = []

    pairing_table = {
        'result idx': [],
        'result mask idx': [],
        'comparison result idx': [],
        'comparison result mask idx': [],
    }
    for i, result in enumerate(results): #loop through results
        if len(result)<1:
            continue
        for j, mask in enumerate(result.masks.data): #loop through each mask of each result
            for k in range(frames_comparison): #looping through the subsequent frames
                centroid = result.masks.centroids[j]
                size = result.masks.sizes[j]
                compared_frame_idx = i+k+1
                if compared_frame_idx > len(results)-1:
                    continue
                comparison_result = results[compared_frame_idx]
                if len(comparison_result) < 1:
                    continue
                for l, comparison_result_mask in enumerate(comparison_result.masks.data):
                    # print(f"orig: img{result.path.split('\\')[1][:3]}, mask_conf: {result.boxes[j].conf[0]}")
                    # print(f"compare: img{comparison_result.path.split('\\')[1][:3]}, mask_conf: {comparison_result.boxes[l].conf[0]}")

                    # size - make sure the masks are of similar size
                    size_similarity = size/comparison_result.masks.sizes[l]
                    if size_similarity < 1:
                        size_similarity = 1/size_similarity #make sure this is always greater than 1, so lower scores are better

                    # shape
                    shape_score = shape_similarity(mask, comparison_result_mask)

                    # location - make sure the centroid y position is within X pixels of previous
                    y_distance = centroid[1] - comparison_result.masks.centroids[l][1]
                    x_distance = centroid[0] - comparison_result.masks.centroids[l][0]
                    frames_ahead = k+1
                    x_distance_per_frame = abs(x_distance/frames_ahead)
                    match = True
                    if x_distance_per_frame > 75:
                        match = False
                    if x_distance_per_frame < 45:
                        match = False
                    if abs(y_distance) > 150:
                        match = False
                    if shape_score*size_similarity >40:
                        match = False
                    if direction:
                        if direction == 'Left':
                            if x_distance > 0:
                                match = False
                        elif direction == 'Right':
                            if x_distance < 0:
                                match = False
                        else:
                            raise ValueError(f'Direction can not be {direction}, can only be one of Positive or Negative')

                    if determine_direction:
                        if match:
                            true_x_distances.append(x_distance)
                    elif direction:
                        if match:
                            pairing_table['result idx'].append(i)
                            pairing_table['result mask idx'].append(j)
                            pairing_table['comparison result idx'].append(compared_frame_idx)
                            pairing_table['comparison result mask idx'].append(l)


    if determine_direction:
        true_x_distances_arr = np.array(true_x_distances)
        median_true_x_distance = np.median(true_x_distances_arr)
        if median_true_x_distance < 0:
            return 'Left'
        else:
            return 'Right'
    elif direction:
        pairing_table = pd.DataFrame.from_dict(pairing_table)
        return pairing_table


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
    # for result in results:
    #     if len(result) < 1:
    #         continue
    #     result.masks.sizes = []
    #     result.masks.centroids = []
    #     result.masks.ids = []
    #
    #     for mask in result.masks.data:
    #         centroid = mask_centroid(mask)
    #         result.masks.centroids.append(centroid)
    #         result.masks.sizes.append(np.sum(mask.cpu().numpy()))
    # camera_direction = group_masks(results, determine_direction=True)
    # print(f'Determined Camera Direction: {camera_direction}')
    # pairing_table = group_masks(results, direction=camera_direction)
    # pairing_table.to_csv('pairing_table.csv')
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

    for result in results:
        if len(result) < 1:
            continue




    print(groups)















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