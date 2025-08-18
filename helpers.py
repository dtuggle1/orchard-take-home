import numpy as np
import cv2
import os
import random
import shutil
import torch
from PIL import Image
from ultralytics import YOLO
import copy
import pandas as pd

class TrunkDetector:
    """
    Trunk Detector model using transfer learning and Yolov8 as the base model. Yolov8 instance segmentation task.
    Looking at state of the art instance segmentation models, Mask R-CNN and Yolov8-seg both could work for this use
    case. Mask R-CNN could provide more accurate results, but likely at the cost of being more computationally heavy.
    Yolov8 is still highly accurate, but simpler architecture, and given the 1 week constraints of this project and
    the limited compute power, Yolov8-seg was chosen. Additionally the nano version of Yolo ,Yolov8n, was chosen as
    I trained the model using my own compute resources (I did not use colab), and wanted the faster training and
    thus iteration time given the constraints. Additionally, the labels are already in Yolov8 format
    (Yolov5 is forward compatible with Yolov8), thus making Yolov8 a good choice.

    Only 30 training images were available, so consideration had to be taken to train with the limited dataset. The most
    obvious and easy solution was to augment the data, and this was done through the Yolo architecture, and the
    augmentations were done in such a way that would best replicate real world scenarios. In this case, the only
    scenario considered was that this model would be used for the evaluation set and an "unseen test set", so the
    augmentations were considered from the images in the evaluation set. See comments next to the augmentations in the
    model to see the rationales behind each. The augmentations were also output as part of the model and reviewed to
    confirm that they were matching with reality; for example, initially hue and saturation were set to be quite high,
    and this caused the augmented images to have lighting that was very unlikely to be reflected in the test set,
    evaluation set, or unseen test set. There were some augmentations that were added for model robustness (as opposed)
    to just mimicking the test set, as the environments of the trunks can vary. The augmentations were tuned as part
    model tuning.

    With only 30 training images, they were split into a standard 80/20 split into training and testing, resulting in
    only 6 images used for testing. Special considerations were taken as with only 6 images in the test set, the risk
    of the model not being generalizable was high. To counteract this, k-folds cross validation was also considered,
    as cross this would have allowed the model to both train and test on the whole dataset, reducing the risk of
    poor generalization. This was ultimately not implemented due to estimated project scope. K-folds cross validation
    unforuntately isn't natively built into Yolo, but Yolo does provide documentation on how to implement it:
    https://docs.ultralytics.com/guides/kfold-cross-validation/.

    Various hyperparameters were tuned as part of the model tuning process. Given that there were additional filtering
    steps that were available given the domain space of this task, as well as the depth data, the model was tuned for
    high recall to try to capture all the available tree trunks, then post-processing was done to filter out the
    false positives. Accordingly so, the confidence was set to be quite conservative.


    """
    def __init__(self):
        self.model = YOLO('yolov8n-seg.pt')  # initialize self.model to be yolo8 seg

    def train(self):
        self.model.train(
            data='model_config.yml',
            seed=1, # ensure reproducibility
            deterministic=True, # ensure reproducibility
            epochs=600, # set this to be high, but given the patience param, it typically quit well before 600
            warmup_epochs=50, # help prevent overtraining on early training data
            patience=100, # The performance plateaued significantly after around 120 epochs, thus setting the patience to 100 allowed ample epochs for there to be no additional learning before stopping the training.
            conf=0.35,  # set this to be low because we can filter later with the depth data
            iou=0.4,  # The trees are set to be far apart and in the foreground so very unlikely to be overlapping with each other, thus intersection over union was set to be low

            # Augmentations
            degrees=20,  # Trunks are vertical, setting this to too high would cause more horizontal branches and objects to be classified, so this was set to be low
            translate=0.3,  # Set higher than default to help detect partially visible tree trunks, but too low and it would start to detect more branches
            fliplr=0.5,  # tree trunks don't have an orientation horizontally so this was set to be high
            flipud=0,  # tree trunks always come out of the ground, so flipping the images would not make sense
            hsv_h=0.1,  # some hue adjustments to allow for various lighting conditions of the trees
            hsv_s=0.2,  # light saturation adjustments to allow for some color varation of different environments
            hsv_v=0.3,  # allow brightness augmentation as the lighting for the trunks may differ
            scale=0.5,  # only looking for trees in the foreground, so keep this at a minimum, as setting this to be too high would find more branches, thick poles, or trunks in the background
            shear=2,  # camera angles aren't always perfect
            perspective=0,  # similar reasoning as shear

            mosaic=1, # This is beneficial for finding occluded trunks or trunks at the edge of the frame, somewhat similar to translate
            mixup=0.1, # Allows for blended composite images, adding robustness, but was done minimally as it does not reflect real life scenario too much.
            copy_paste=0.3, # Allows for more training label instances in different environments

            # Learning params
            lr0=0.01,  # set initial learning rate to high such that it converges faster
            lrf=0.01,  # set final learning rate to 1/100 of lr0 for finer tuning and preventing overshoot
            plots=True,

            save_period=50

        )

def empty_and_create_folder(folder_name):
    """
    Given a folder name, deletes it and all of its content if it exists, then remakes it
    """
    if folder_name in os.listdir():
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)

def get_depth_filename_from_image_filename(image_filename):
    """
    Returns the depth filename given the image filename
    """
    return f"{image_filename[:3]}_depth.png"

def get_label_filename_from_training_image_filename(image_filename):
    """
    Returns the label filename given the training image filename
    """
    return f"{image_filename[:3]}.txt"

def convert_label_contour(label_line, img_shape):
    """
    GENERATED BY AI
    Creates a contour of a line from a label file for easy overlay onto an image
    """
    parts = label_line.strip().split()
    class_label = int(parts[0])
    if class_label != 0:
        return None
    coordinates = [(float(parts[i]) * img_shape[1], float(parts[i + 1]) * img_shape[0]) for i in range(1, len(parts), 2)]
    return np.int32([coordinates])

def overlay_label(label_file_path, image, transparency_level):
    """
    Overlays a single label in translucent red on an image and returns the image
    """
    with open(label_file_path, 'r') as file:
        for line in file:
            contour = convert_label_contour(line, image.shape)
            if contour is not None:
                cv2.fillPoly(image, [contour], (0, 0, 255))

    masked_image = cv2.addWeighted(image, transparency_level, image,
                                   1 - transparency_level,
                                   0)  # Asked CHATGPT how to create translucent

    return masked_image

def split_training_data(yolo_train_data_folder, training_data_folder, percentage_to_train):
    """
    Given the original training data folder structure, randomly splits the training images into a train and test set
    given a percentage, creates the appropriate file structure that yolo expects, and adds the images and labels to the
    corresponding folders.
    """
    empty_and_create_folder(yolo_train_data_folder)
    os.mkdir(yolo_train_data_folder)
    os.mkdir(os.path.join(yolo_train_data_folder, 'train'))
    os.mkdir(os.path.join(yolo_train_data_folder, 'train', 'images'))
    os.mkdir(os.path.join(yolo_train_data_folder, 'train', 'labels'))
    os.mkdir(os.path.join(yolo_train_data_folder, 'val'))
    os.mkdir(os.path.join(yolo_train_data_folder, 'val', 'images'))
    os.mkdir(os.path.join(yolo_train_data_folder, 'val', 'labels'))

    image_files = os.listdir(os.path.join(training_data_folder, 'Images'))
    training_length = int(len(image_files)*percentage_to_train)
    training_files = random.sample(image_files, training_length)
    for training_file in training_files: #Training Images
        shutil.copy(os.path.join(training_data_folder, 'Images', training_file),
                    os.path.join(yolo_train_data_folder,'train','images'))

        shutil.copy(os.path.join(training_data_folder, 'Labels', get_label_filename_from_training_image_filename(training_file)),
                    os.path.join(yolo_train_data_folder,'train','labels'))
    for training_file in training_files:
        image_files.remove(training_file)
    for image_file in image_files: # Validation Images
        shutil.copy(os.path.join(training_data_folder, 'Images', image_file),
                    os.path.join(yolo_train_data_folder,'val','images'))
        shutil.copy(os.path.join(training_data_folder, 'Labels', get_label_filename_from_training_image_filename(image_file)),
                    os.path.join(yolo_train_data_folder,'val','labels'))

def overlay_labels(image_folder, label_folder, output_folder):
    """
    Overlay the Labels on the training image for easier viewing
    """
    for image_file in os.listdir(image_folder):
        training_image_path = os.path.join(image_folder, image_file)
        training_image = cv2.imread(training_image_path)
        training_image_label_filename = get_label_filename_from_training_image_filename(
            image_file)
        training_image_label_path = os.path.join(label_folder, training_image_label_filename)
        masked_image = overlay_label(training_image_label_path, training_image, 0.4)
        # Save the result
        cv2.imwrite(os.path.join(output_folder, f"{image_file[:3]}_masked.jpg"),
                                 masked_image)

def mask_overlaps(result):
    """
    AI WROTE THIS FUNCTION. I asked it to tell me the percentage overlap between masks using the pairwise Intersection
    Over Union. This function provided the percentage overlap between the masks of a yolo result object, which can
    then be used to group the masks together based on which makes have a high amount of overlap.


    Returns:
      iou: [N,N] IoU matrix between masks
      overlap_smaller: [N,N] intersection / min(area_i, area_j)
      ids: list of indices for masks (0..N-1)
    """
    if result.masks is None or len(result.masks) <= 1:
        return None, None, []

    # result.masks.data: [N, H, W] (float/byte on device)
    m = result.masks.data
    if m.dtype != torch.bool:
        m = m > 0.5  # binarize just in case

    # areas: [N]
    areas = m.flatten(1).sum(dim=1).float()

    # intersections: [N,N]
    # (broadcasted AND; use multiply for speed on GPU)
    inter = (m.unsqueeze(1) & m.unsqueeze(0)).flatten(2).sum(dim=2).float()

    # unions: [N,N]
    unions = areas.unsqueeze(1) + areas.unsqueeze(0) - inter
    iou = torch.where(unions > 0, inter / unions, torch.zeros_like(unions))

    # percent overlap relative to the smaller mask
    mins = torch.minimum(areas.unsqueeze(1), areas.unsqueeze(0))
    overlap_smaller = torch.where(mins > 0, inter / mins, torch.zeros_like(mins))

    return iou.cpu().numpy(), overlap_smaller.cpu().numpy(), list(range(m.shape[0]))

def group_from_overlap(over_small, thresh=0.7):
    """
    AI WROTE THIS FUNCTION. I asked it to group the masks by the over_small array (the output from the masks_overlap
    function). It groups the masks together if they are overlapping more than the specified threshold. These groups
     get fed into the next part of the code, which selects the highest confidence mask of the group, and only keeps that
     mask. This is needed as part of the filtering process to filter out masks that are overlapping.

    over_small: 2D numpy array from mask_overlaps()
    thresh: overlap threshold (0.5 = 50%)
    Returns: list of lists, where each inner list is a group of indices.
    """
    n = over_small.shape[0]
    visited = [False] * n
    groups = []

    for i in range(n):
        if not visited[i]:
            group = [i]
            visited[i] = True
            # check others
            for j in range(n):
                if not visited[j] and over_small[i, j] >= thresh:
                    group.append(j)
                    visited[j] = True
            groups.append(group)

    return groups

def remove_overlaps(result):
    """
    Removes overlapping masks.
    First, determines the percentage overlap of each mask of the result.
    Then it groups the masks based on which masks are largely overlapping with each other.
    From each group, it selects and keeps only the mask with the highest confidence, and puts that back into the
    results object.
    """
    if len(result)<=1:
        return result
    overlaps = mask_overlaps(result) # Determine overlaps of the masks
    mask_groups = group_from_overlap(overlaps[1]) # Group the masks by the overlaps
    best_masks = [] # Select the best mask of each group
    for mask_group in mask_groups:
        confidences = []
        for mask in mask_group:
            confidences.append((mask, result.boxes.conf[mask]))
        best_mask_index, best_max_confidence = max(confidences, key=lambda x: x[1])  # LINE WRITTEN BY AI - I asked it to write me a line of code that picks a tuple based on the maximum value of one elemtn of the tuple
        best_masks.append(best_mask_index)
        # take the mask with the highest confidence
    result.masks = result.masks[best_masks]
    result.boxes = result.boxes[best_masks]
    return result

def apply_masks_to_depth_per_instance(result, depth_img):
    """
    Returns a list of masked depth images, one per instance.
    Pixels outside each mask are set to invalid_value.
    """
    if result.masks is None:
        return []

    depth_masks = []
    for pts in result.masks.xy:  # pixel coordinates in original size
        mask = np.zeros(depth_img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)

        masked_depth = depth_img.copy()
        masked_depth[mask == 0] = 255
        depth_masks.append(masked_depth)

    return depth_masks

def shape_similarity(mask1, mask2):
    """
    FUNCTION WRITTEN BY AI. Asked it to build me a metric that compares shapes of two masks
    """
    mask1 = mask1.cpu().numpy()
    mask2 = mask2.cpu().numpy()
    c1, _ = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c2, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not c1 or not c2:
        return None

    return cv2.matchShapes(c1[0], c2[0], cv2.CONTOURS_MATCH_I1, 0.0)

def mask_centroid(mask):
    """
    FUNCTION WRITTEN BY AI

    Calculates the centroid of a mask and returns the centroid coordinates.
    """
    # mask should be 2D, with 1s where the object is
    mask = mask.detach().cpu().numpy()
    ys, xs = np.nonzero(mask)  # get coordinates of all positive pixels
    if len(xs) == 0:  # avoid division by zero
        return None
    cx = np.mean(xs)
    cy = np.mean(ys)
    return (cx, cy)

def get_depth_mask(mask, depth_img,resize=False):
    """
    Given a depth image and a mask, returns the masked depth image
    """

    binary_mask = mask.cpu().numpy()
    if resize:
        binary_mask = cv2.resize(binary_mask, (1200, 1920), interpolation=cv2.INTER_NEAREST)

    masked_depth = depth_img.astype(float)
    masked_depth[binary_mask == 0] = np.nan
    masked_depth = masked_depth[:, :, 0]
    return masked_depth

def calc_depth_mask_metrics(masked_depth):
    metrics = {
        'median': np.nanmedian(masked_depth),
        'p1': np.nanpercentile(masked_depth, 1),
        'p25': np.nanpercentile(masked_depth, 25),
        'p75': np.nanpercentile(masked_depth, 75),
        'p99': np.nanpercentile(masked_depth, 99),
        'mean': np.nanmean(masked_depth),
        'std': np.nanstd(masked_depth),
        'max': np.nanmax(masked_depth),
        'min': np.nanmin(masked_depth)
    }
    return metrics

def filter_by_depth(result, evaluation_set_depth_folder, image_filename):
    """
    Filters the false positives out using the depth data. It determines the range of the 1st and 99th percentile
    of the depth points, and keeps only the masks that fill within a certain range.

    Note this is ultimately not used.
    """
    depth_image_filename = get_depth_filename_from_image_filename(image_filename)
    depth_img = cv2.imread(os.path.join(evaluation_set_depth_folder, depth_image_filename))
    if result.masks is not None:
        keep_mask_indeces = []
        for i, mask in enumerate(result.masks.data):
            masked_depth = get_depth_mask(mask, depth_img,resize=False)
            masked_depth_metrics = calc_depth_mask_metrics(masked_depth)
            conf = result.boxes.conf[i]
            # print(f"{image_filename},{i},{conf},{median},{mean},{std},{max},{min},{p1},{p25},{p75},{p99}")
            p99_p1_range = masked_depth_metrics['p99'] - masked_depth_metrics['p1']
            # cv2.imwrite('masked_depth.png', masked_depth)
            if (conf > 0.5) or (p99_p1_range > 7):
                keep_mask_indeces.append(i)
        result.masks = result.masks[keep_mask_indeces]
        result.boxes = result.boxes[keep_mask_indeces]
    return result

def remove_nearby_predictions(result):
    """
    For masks that appear within 100 pixels of each other horizontally, remove all but the highest confidence mask.
    """
    if result.masks is not None:
        keep_mask_indeces = list(range(len(result.masks.data)))
        cxs = []
        for i, mask in enumerate(result.masks.data):
            cx, cy = mask_centroid(mask)
            cxs.append(cx)
        if len(cxs) > 1:
            indeces_to_remove = []
            for i, cx in enumerate(cxs):
                for j in range(i+1,len(cxs)):
                    if abs(cx-cxs[j]) < 100:
                        conf_i = result.boxes.conf[i]
                        conf_j = result.boxes.conf[j]
                        if conf_i > conf_j:
                            indeces_to_remove.append(j)
                        else:
                            indeces_to_remove.append(i)
            for index_to_remove in indeces_to_remove:
                if index_to_remove in keep_mask_indeces:
                    keep_mask_indeces.remove(index_to_remove)

        result.masks = result.masks[keep_mask_indeces]
        result.boxes = result.boxes[keep_mask_indeces]
    return result

def process_images(model, images, depth_folder_name=None, depth_filtering=False, save_folder=None):
    """
    The function executes the full processing of an a collection of images, running it through the model, then
    post-processing. The model was tuned to achieve as high of recall as possible as there was significant data to
    post-process the image given the nature of the task. We know that the trunks are on a farm that are spaced apart,
    and that we are only looking in the foreground of the scene, so the trees should not be overlapping with each other,
    thus we can remove overlapping masks, keeping the mask with the highest confidence value. Additionally, because
    the trunks are spaced apart, if the predictions are too near to each other, there must be a false positive as the
    trees cannot exist too close to each other, again, keeping the mask with the highest confidence value.

    Filtering by depth was considered because true positives were likely to have more defined depth values than predictions
    that were looking into open space or a mixture of near and far, but after exploring the depth data of both true positives
    and false positives, a simple heuristic to filter the false positives could not be determined. The leading candidate
    for a heuristic was to look at the range of depth values of the mask between the 99th and 1st percentile (not the
    true max and mins to filter out outliers), as it was theorized and observed that the true positives have a smaller
    range than the false positives, as the depth of the trunk would only vary by the diameter of the trunk, but the
    depth of a false positive can be empty space and leaves and would vary largely. In a broader project, I would have
    trained a binary classification convolutional neural network on the depth masks.

    """
    results = model(images, conf=0.35) # run the model
    output_images = []
    output_results = []
    if save_folder:
        empty_and_create_folder(save_folder)
    for result in results:

        # Remove overlaps
        result = remove_overlaps(result)

        # Remove predictions that are close to each other
        result = remove_nearby_predictions(result)


        # Filter by depth - ULTIMATELY NOT USED BUT KEPT IN PLACE
        if depth_filtering:
            image_filename = result.path.split('\\')[1] #clean up
            result = filter_by_depth(result, depth_folder_name, image_filename)

        im = result.plot(txt_color=(0, 0, 255))
        if save_folder:
            image_filename = result.path.split('\\')[1] #clean up
            save_path = os.path.join(save_folder, f"{image_filename[:3]}_predict.jpg")
            Image.fromarray(im).save(save_path)
        output_images.append(copy.deepcopy(im))
        output_results.append(copy.deepcopy(result))
    return output_images, output_results

def generate_combined_mask(result):
    """
    Combines the masks of the result, and outputs a singular mask that can be overlayed on an image
    """
    masks = result.masks.data.cpu().numpy()
    combined_mask = np.max(masks, axis=0).astype(np.uint8)
    combined_mask = combined_mask * 255
    return combined_mask

def calculate_mask_similarities(results, direction=None, determine_direction=False):
    """
    Outputs a pairing table indicating which masks are a match and which are not. Only the masks in the subsequent 8
    frames are evaluated.

    If direction is one of Left or Right, output the final outputs, using Left or Right to filter out masks that are
    not in the appropriate direction.
    If determine_direction is True, run a rough matching process (without the direction), and the median direction
    of the True labels will determine the direction of the cameras.



    """
    if direction and determine_direction:
        raise ValueError('Cannot both determine the direction and use direction for final outputs')
    if not direction and not determine_direction:
        raise ValueError('Must select at least one of direction or determine_direction')
    frames_comparison = 8
    true_x_distances = []

    pairing_table = {
        'result idx': [],
        'result mask idx': [],
        'comparison result idx': [],
        'comparison result mask idx': [],
    }
    debug_table = {
        'result idx': [],
        'result mask idx': [],
        'comparison result idx': [],
        'comparison result mask idx': [],
        'x distance': [],
        'y distance': [],
        'x distance per frame': [],
        'shape * similarity': [],
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

                    debug_table['result idx'].append(i)
                    debug_table['result mask idx'].append(j)
                    debug_table['comparison result idx'].append(compared_frame_idx)
                    debug_table['comparison result mask idx'].append(l)
                    debug_table['x distance'].append(x_distance)
                    debug_table['y distance'].append(y_distance)
                    debug_table['x distance per frame'].append(x_distance_per_frame)
                    debug_table['shape * similarity'].append(shape_score*size_similarity)

                    match = True
                    if x_distance_per_frame > 80:
                        match = False
                    if x_distance_per_frame < 40:
                        match = False
                    if abs(y_distance) > 150:
                        match = False
                    if shape_score*size_similarity >50:
                        match = False
                    if direction:
                        if direction == 'Left':
                            if x_distance > 0:
                                match = False
                        elif direction == 'Right':
                            if x_distance < 0:
                                match = False
                        else:
                            raise ValueError(f'Direction can not be {direction}, can only be one of Left or Right')

                    if determine_direction:
                        if match:
                            true_x_distances.append(x_distance)
                    elif direction:
                        if match:
                            pairing_table['result idx'].append(i)
                            pairing_table['result mask idx'].append(j)
                            pairing_table['comparison result idx'].append(compared_frame_idx)
                            pairing_table['comparison result mask idx'].append(l)
    debug_table_df = pd.DataFrame.from_dict(debug_table)
    debug_table_df.to_csv('debug_table.csv')


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

def place_label_simple(image, text, x, y, font, font_scale, thickness, bg_color=(255, 255, 255),
                       text_color=(0, 0, 0)):
    """
    FUNCTION WRITTEN BY AI
    Simple label placement that ensures text stays in rectangle
    """
    img_h, img_w = image.shape[:2]

    # Get text dimensions
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    padding = 5
    total_width = text_width + 2 * padding
    total_height = text_height + 2 * padding

    # Default position (above the object)
    label_x = x
    label_y = y - 10

    # Adjust if going off edges
    if label_x + total_width > img_w:
        label_x = img_w - total_width
    if label_x < 0:
        label_x = 0

    if label_y - total_height < 0:
        label_y = y + 40  # Place below instead

    if label_y > img_h:
        label_y = img_h - 10

    # Calculate background rectangle
    bg_x1 = label_x
    bg_y1 = label_y - text_height - padding
    bg_x2 = label_x + text_width + 2 * padding
    bg_y2 = label_y + padding

    # Ensure background stays in bounds
    bg_x1 = max(0, min(bg_x1, img_w - total_width))
    bg_x2 = min(img_w, bg_x1 + total_width)
    bg_y1 = max(0, min(bg_y1, img_h - total_height))
    bg_y2 = min(img_h, bg_y1 + total_height)

    # Draw background
    cv2.rectangle(image, (int(bg_x1), int(bg_y1)), (int(bg_x2), int(bg_y2)), bg_color, -1)

    # Draw text centered in the background
    text_x = bg_x1 + padding
    text_y = bg_y1 + text_height + padding
    cv2.putText(image, text, (int(text_x), int(text_y)), font, font_scale, text_color, thickness)

def place_label_safely(image, text, x, y, font, font_scale, thickness, bg_color=(255, 255, 255),
                       text_color=(0, 0, 0)):
    """
    FUNCTION WRITTEN BY AI
    Place a label with background that stays within image bounds
    """
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

def output_p4(results, class_labels, display_images=True, output_folder_name=False):
    """
    Given the results of the yolo output and the class labels, generates the images with the overlaid masks, then
    optionally outputs and/or displays them
    """
    if output_folder_name:
        empty_and_create_folder(output_folder_name)
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

        if output_folder_name:
            cv2.imwrite(os.path.join(output_folder_name, f"{result.path.split('\\')[1][:3]}_output.jpg"),output_image)

def create_groups_from_pairing_table(pairing_table):
    """
    Given the pairing table of masks, create the groups of similar masks that are to be identified as the same trunk.
    """
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
    return groups