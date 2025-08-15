import numpy as np
import cv2
import os
import random
import shutil

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






