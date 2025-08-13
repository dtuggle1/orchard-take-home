from helpers import *
import cv2
import os

IMAGE_INDEX = 25

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
depth_at_pixel = depth_image[Y_INDEX][X_INDEX]
print(f"The depth of this same pixel, via the depth map, returned in millimeters: {depth_at_pixel}")

# The total number of pixels in this image that have depth values +/- 10 mm of the depth value you found.
BOUND_PLUS_MINUS = 10 # bounding in mm
mask = (depth_image > depth_at_pixel-BOUND_PLUS_MINUS) & (depth_image < depth_at_pixel+BOUND_PLUS_MINUS)
print(f"The total number of pixels in this image that have depth values +/- 10 mm of the depth value you found: {sum(mask)}")