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
            degrees=20,  # how much it is anticipated to see trees with differing levels of rotation
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

def get_depth_filename_from_image_filename(image_filename):
  return f"{image_filename[:3]}_depth.png"

def get_label_filename_from_training_image_filename(image_filename):
    return f"{image_filename[:3]}.txt"

def convert_label_contour(label_line, img_shape):
    parts = label_line.strip().split()
    class_label = int(parts[0])
    if class_label != 0:
        return None
    coordinates = [(float(parts[i]) * img_shape[1], float(parts[i + 1]) * img_shape[0]) for i in range(1, len(parts), 2)]
    return np.int32([coordinates])

def overlay_label(label_file_path, image, transparency_level):
    with open(label_file_path, 'r') as file:
        for line in file:
            contour = convert_label_contour(line, image.shape)
            if contour is not None:
                cv2.fillPoly(image, [contour], (0, 0, 255))

    masked_image = cv2.addWeighted(image, transparency_level, image,
                                   1 - transparency_level,
                                   0)  # Asked CHATGPT how to create translucent

    return masked_image

def split_training_data(percentage_to_train):
    if 'YoloTrainData' in os.listdir():
        shutil.rmtree('YoloTrainData')
    os.mkdir('YoloTrainData')
    os.mkdir(os.path.join('YoloTrainData','train'))
    os.mkdir(os.path.join('YoloTrainData','train','images'))
    os.mkdir(os.path.join('YoloTrainData','train','labels'))
    os.mkdir(os.path.join('YoloTrainData','val'))
    os.mkdir(os.path.join('YoloTrainData','val','images'))
    os.mkdir(os.path.join('YoloTrainData','val','labels'))

    image_files = os.listdir(os.path.join('Training Data', 'Images'))
    training_length = int(len(image_files)*percentage_to_train)
    training_files = random.sample(image_files, training_length)
    for training_file in training_files: #Training Images
        shutil.copy(os.path.join('Training Data', 'Images', training_file),
                    os.path.join('YoloTrainData','train','images'))

        shutil.copy(os.path.join('Training Data', 'Labels', get_label_filename_from_training_image_filename(training_file)),
                    os.path.join('YoloTrainData','train','labels'))
    for training_file in training_files:
        image_files.remove(training_file)
    for image_file in image_files: # Validation Images
        shutil.copy(os.path.join('Training Data', 'Images', image_file),
                    os.path.join('YoloTrainData','val','images'))
        shutil.copy(os.path.join('Training Data', 'Labels', get_label_filename_from_training_image_filename(image_file)),
                    os.path.join('YoloTrainData','val','labels'))

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
    AI WROTE THIS FUNCTION. I asked it to tell me the percentage overlap between masks

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
    AI WROTE THIS FUNCTION. I asked it to group the masks by the over_small array

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
    if len(result)<=1:
        return result
    overlaps = mask_overlaps(result)
    mask_groups = group_from_overlap(overlaps[1])
    best_masks = []
    for mask_group in mask_groups:
        confidences = []
        for mask in mask_group:
            confidences.append((mask, result.boxes.conf[mask]))
        best_mask_index, best_max_confidence = max(confidences, key=lambda x: x[1])  # WRITTEN BY AI
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
    # FUNCTION WRITTEN BY AI. Asked to build me a metric that compares shapes of two masks
    mask1 = mask1.cpu().numpy()
    mask2 = mask2.cpu().numpy()
    c1, _ = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c2, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not c1 or not c2:
        return None

    return cv2.matchShapes(c1[0], c2[0], cv2.CONTOURS_MATCH_I1, 0.0)

def mask_centroid(mask):
    # FUNCTION WRITTEN BY AI
    # mask should be 2D, with 1s where the object is
    mask = mask.detach().cpu().numpy()
    ys, xs = np.nonzero(mask)  # get coordinates of all positive pixels
    if len(xs) == 0:  # avoid division by zero
        return None
    cx = np.mean(xs)
    cy = np.mean(ys)
    return (cx, cy)

def get_depth_mask(mask, depth_img,resize=False):
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
    results = model(images, conf=0.35)
    output_images = []
    output_results = []
    for result in results:

        # Remove overlaps
        result = remove_overlaps(result)

        # Remove predictions that are close to each other
        result = remove_nearby_predictions(result)


        # Filter by depth
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
    masks = result.masks.data.cpu().numpy()
    combined_mask = np.max(masks, axis=0).astype(np.uint8)
    combined_mask = combined_mask * 255
    return combined_mask


def group_masks(results, direction=None, determine_direction=False):
    """
    if direction is True, output the final outputs
    if determine_direction is True, run a rough matching process (without the direction), and the median direction of the True labels will determine the direction of the cameras
    """
    if direction and determine_direction:
        raise ValueError('Cannot both determine the direction and use direction for final outputs')
    if not direction and not determine_direction:
        raise ValueError('Must be ')
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
    Simple label placement that ensures text stays in rectangle"""
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