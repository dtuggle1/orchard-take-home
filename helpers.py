import numpy as np
import cv2
import os
import random
import shutil
import torch

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

def get_depth_from_mask(image, contours):

    pass

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

def remove_overlaps(result):
    pass

def filter_by_depth(result):
    pass

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