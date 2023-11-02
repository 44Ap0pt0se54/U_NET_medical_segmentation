import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import zipfile
import cv2
import os

class Model:

    def __init__(self):

        self.model = tf.keras.models.load_model("/content/Unet_3L")
       
    def test_model(self):

        with zipfile.ZipFile("/content/test_image.zip", "r") as zip_ref:
            zip_ref.extractall("/content/test_images_folder")
        with zipfile.ZipFile("/content/test_mask.zip", "r") as zip_ref:
            zip_ref.extractall("/content/test_mask_folder")

        image_folder = '/content/test_images_folder/image'

        # Get a list of image file names in the folder
        image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith('.png')]

        # Initialize an empty list to store the image data
        image_data = []

        # Loop through the image files and convert them to NumPy arrays
        for image_file in image_files:
            img = cv2.imread(image_file)
            if img is not None:
                resized_image = cv2.resize(img, (128, 128))
                grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                image_data.append(grayscale_image)

        # Convert the list of images to a NumPy array
        test_images = np.array(image_data)

        ################################################################################################

        image_folder = '/content/test_mask_folder/mask'

        # Get a list of image file names in the folder
        image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith('.png')]

        # Initialize an empty list to store the image data
        image_data = []

        # Loop through the image files and convert them to NumPy arrays
        for image_file in image_files:
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                resized_image = cv2.resize(img, (128, 128))
                image_data.append(resized_image)

        # Convert the list of images to a NumPy array
        test_masks = np.array(image_data)

        test_images = (test_images - np.min(test_images)) / (np.max(test_images) - np.min(test_images))
                                                                                                        # Min Max normalization
        test_masks = (test_masks - np.min(test_masks)) / (np.max(test_masks) - np.min(test_masks))

        for k in range(len(test_masks)):              # Thresholding for test masks in order to compute Dice & IoU
            test_masks[k][ test_masks[k] > 0] = 1

        pred = self.model.predict(test_images)

        for i in range(len(pred)):
            pred[i][pred[i] > thresh] = 1
            pred[i][pred[i] <= thresh] = 0

        mask1_flat = pred.flatten()
        mask2_flat = test_masks.flatten()
        intersection = np.sum(mask1_flat * mask2_flat)
        mask1_sum = np.sum(mask1_flat)
        mask2_sum = np.sum(mask2_flat)
        dice = (2.0 * intersection) / (mask1_sum + mask2_sum)
        union = np.sum(mask1_flat) + np.sum(mask2_flat) - intersection
        iou = intersection / union

        print("Dice : ", dice)
        print("IoU : ", IoU)


