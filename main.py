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
TRAINING_DATA_FOLDER_NAME = 'Training Data'
MASKED_TRAINING_DATA_FOLDER_NAME = 'MaskedTrainingData'
CONF_THRESH = .35

EVALUATE_EVALUATION_SET = True
OVERLAY_LABELS_ON_TRAIN_DATA = True

BEST_MODEL = os.path.join('runs','segment','train35','weights','best.pt')
TRAIN_MODEL = True

class TrunkDetector:

    def __init__(self):
        self.model = YOLO('yolov8n-seg.pt')  # initialize self.model to be yolo8 seg

    def train(self):
        self.model.train(
            data='model_config.yml',
            seed=1,
            deterministic=True,
            epochs=600,
            warmup_epochs=50,
            patience=100,
            conf=0.1,  # set this to be low because we can filter later with the depth data
            iou=0.4,  # Help prevent overlaps

            # Augmentations
            degrees=10,  # how much it is anticipated to see trees with differing levels of rotation
            translate=0.3,  # help with detecting partially visible tree trunks
            fliplr=0.5,  # trees dont have a left/right orientation so adding this provides more good data
            flipud=0,  # tree trunks always have a certain orientation coming out of the ground
            hsv_h=0.1,  # allow for default hue adjustment because the lighting for the trees can vary
            hsv_s=0.2,  # hu
            hsv_v=0.3,  # allow brightness augmentation
            scale=0.5,  # only looking for trees in the foreground, so keep this at a minimum
            shear=2,  # camera angles aren't always perfect
            perspective=0,  # similar reasoning as shear

            mosaic=1,
            mixup=0.1,
            copy_paste=0.3,

            # Learning params
            lr0=0.01,  # set initial learning rate to high such that it converges faster
            lrf=0.01,  # set final learning rate to 1/100 of lr0 for finer tuning and preventing overshoot
            plots=True,

            save_period=50

        )


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

    # for file in os.listdir(os.path.join('Training Data', 'Images')):
    #     if '_rgb' in file:
    #         new_filename = f"{file[:3]}.jpg"
    #         os.rename(os.path.join('Training Data', 'Images', file), os.path.join('Training Data', 'Images', new_filename))
    # for file in os.listdir(os.path.join('Training Data', 'Labels')):
    #     if '_label' in file:
    #         new_filename = f"{file[:3]}.txt"
    #         os.rename(os.path.join('Training Data', 'Labels', file), os.path.join('Training Data', 'Labels', new_filename))




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

    if EVALUATE_EVALUATION_SET:
        if EVALUATION_SET_OUTPUTS_FOLDER_NAME in os.listdir():
            shutil.rmtree(EVALUATION_SET_OUTPUTS_FOLDER_NAME)
        os.mkdir(EVALUATION_SET_OUTPUTS_FOLDER_NAME)

        eval_images = []
        # for eval_image_filename in os.listdir(EVALUATION_SET_RGB_FOLDER_NAME)[:1]: #TODO: REMOVE, ONLY 3 FOR TESTING
        # for eval_image_filename in os.listdir(EVALUATION_SET_RGB_FOLDER_NAME)[16:17]: #TODO: REMOVE, ONLY 3 FOR TESTING
        for eval_image_filename in os.listdir(EVALUATION_SET_RGB_FOLDER_NAME): #TODO: REMOVE, ONLY 3 FOR TESTING
            eval_images.append(os.path.join(EVALUATION_SET_RGB_FOLDER_NAME, eval_image_filename))


        results = model(eval_images, conf=CONF_THRESH) #todo: refactor
        for result in results:
            image_filename = result.path.split('\\')[1]

            # Remove overlaps
            result = remove_overlaps(result)


            # Filter by depth
            depth_image_filename = get_depth_filename_from_image_filename(image_filename)
            depth_img = cv2.imread(os.path.join(EVALUATION_SET_DEPTH_FOLDER_NAME, depth_image_filename))
            if result.masks is not None:
                for mask in result.masks.data:
                    binary_mask = mask.cpu().numpy()
                    binary_mask = cv2.resize(binary_mask, (1200,1920), interpolation=cv2.INTER_NEAREST)
                    mask_out = (binary_mask * 255).astype('uint8')
                    cv2.imwrite('test.png', mask_out)

                    masked_depth = depth_img.astype(np.int32)
                    masked_depth[binary_mask==0] = -1
                    print()

                # depth_masks = apply_masks_to_depth_per_instance(result, depth_img)
                # for depth_mask in depth_masks:
                #     cv2.imshow("Masked Depth", depth_mask)




            im = result.plot(txt_color=(0,0,255))
            save_path = os.path.join(EVALUATION_SET_OUTPUTS_FOLDER_NAME, f"{image_filename[:3]}_predict.jpg")
            Image.fromarray(im).save(save_path)
            print()


    # Things I would do with more time
    # Detect ground/soil and verify that the the bottom of trunks are embedded in the
    # ground.
    return model

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
    original_image = cv2.imread(BytesIO(response.content))


    ### Run inference using your model here ###
    ml_inference_output = original_image
    # ml_inference_output should be the pixel-wise mask, with each pixel either (0, 0, 0) or (255, 255, 255) depending on whether it is part of a trunk mask
    model.predict(ml_inference_output)

    # Save the image
    ml_inference_output.save('test_output_image.jpg')
    print("ML Inference Output Image:")
    cv2.show(ml_inference_output)

    # Save the image
    ml_inference_output.save('ml_inference_output.jpg')

def part4():
    """
    Part 4

    In this part, you will need to develop a method to count the total number of trunks in the sequence of Evaluation Set Images.

    A portion of this will involve using your trained ML model to inference on the RGB Images of the trunks in the EvaluationSetRGB folder, to detect the trunks.

    Afterwards, your method must be able to count the number of unique trunks in the sequence of Images, without undercounting or double-counting any given trunk.

    The result of this part should be one number, indicating the total number of trunks in the sequence of 49 Images.

    In addition, output the inferences of your ML model on each of the 49 Images, with the tree mask visible in translucent red, plotted on top of the original RGB image. On top of each mask, plot text indicating the unique ID of each tree. (i.e. if there are 20 unique trees in the sequence of 49 Images, the first 4 Images might all contain tree 1, the 5th image might contain tree 1 and tree 2, the next 5 Images might only contain tree 2, etc. Plot the unique ID assigned to each tree on top of that tree's mask).
    """
    # Count trunks here



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
    # split_training_data(0.8)
    part2()
    # part3()
    # print("torch:", torch.__version__)
    # print("CUDA available:", torch.cuda.is_available())
    # if torch.cuda.is_available():
    #     print("Device:", torch.cuda.get_device_name(0))
    # part4()
    # part5()