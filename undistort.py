import cv2
import numpy as np
import os
import glob

# Load the calibration parameters
calib_data = np.load('calibration.npz')
mtx = calib_data['mtx']
dist = calib_data['dist']

# Create the output directory if it doesn't exist
output_dir = 'undistorted_images'
os.makedirs(output_dir, exist_ok=True)

# Get the list of images
image_files = glob.glob('converted_images/*.jpg')

# Process each image
for image_file in image_files:
    # Load the image
    img = cv2.imread(image_file)
    h, w = img.shape[:2]

    # Get the optimal camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Undistort the image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Crop the image based on the region of interest (ROI)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # Construct the output file path
    output_file = os.path.join(output_dir, os.path.basename(image_file))

    # Save the undistorted image
    cv2.imwrite(output_file, dst)

print("All images have been undistorted and saved to the 'undistorted_images' directory.")
