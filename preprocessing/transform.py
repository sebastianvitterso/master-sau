"""
Process for transforming raw images (RGB and IR), as well as (YOLO-)labels 
corresponding to the raw RGB images, into images (still RGB and IR) of the same size,
and making the labels fit the new (slightly cropped) images.

Depends on research done by Kari Meling Johannesen in her masters: https://github.com/thaiKari/sheepDetector
"""
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from Label import LabelSet
from Image import Image

# base folders
INPUT_BASE_FOLDER = './data/raw/'
CROPPED_BASE_FOLDER = './data/cropped/'
PARTITION_BASE_FOLDER = './data/partitioned/'

# base folder structure
RGB_FOLDER = 'rgb/'
IR_FOLDER = 'ir/'
LABEL_FOLDER = 'labels/'

def show_blended_output(filename:str, partition:tuple[int, int]=None):
    '''@param filename The filename without the filetype/ending. E.g. `2021_09_holtan_0535`'''

    base_folder = CROPPED_BASE_FOLDER
    if(partition is not None):
        base_folder = PARTITION_BASE_FOLDER
        filename = f"{filename}_p{partition[0]}{partition[1]}"

    rgb_image = Image.loadFromImagePath(base_folder + RGB_FOLDER + filename + ".JPG", is_cropped=True)
    # ir_image = Image.loadFromImagePath(base_folder + IR_FOLDER + filename + ".JPG", is_cropped=True)
    label_set = LabelSet.loadFromFilePath(base_folder + LABEL_FOLDER + filename + ".txt", is_cropped=True)

    # blended_img = np.maximum(rgb_image.img, ir_image.img)
    blended_img = rgb_image.img

    for label in label_set.labels:
        blended_img = cv2.rectangle(blended_img, (label.left, label.top), (label.right, label.bottom), (0,0,255), 2)
    
    plt.figure()
    # plt.subplot(2, 3, 1)
    plt.subplot(2, 3, (1, 4))
    plt.imshow(rgb_image.img)
    # plt.subplot(2, 3, 4)
    # plt.imshow(ir_image.img)
    plt.subplot(2, 3, (2,6))
    plt.imshow(blended_img)
    # plt.get_current_fig_manager().full_screen_toggle()
    plt.show()


def transform_data():
    for filename in os.listdir(INPUT_BASE_FOLDER + RGB_FOLDER):
        image = Image.loadFromImagePath(INPUT_BASE_FOLDER + RGB_FOLDER + filename, is_cropped=False)
        # image = image.crop()
        # image.saveToImagePath(CROPPED_BASE_FOLDER + RGB_FOLDER + filename)
        fileroot = filename.split('.')[0]
        for partition_image in image.partitions():
            partition_filename = f"{fileroot}_p{partition_image.partition_coordinates[0]}{partition_image.partition_coordinates[1]}.JPG"
            partition_image.saveToImagePath(PARTITION_BASE_FOLDER + RGB_FOLDER + partition_filename)
        print(f"Processed RGB: {fileroot}")

    # for filename in os.listdir(INPUT_BASE_FOLDER + IR_FOLDER):
    #     image = Image.loadFromImagePath(INPUT_BASE_FOLDER + IR_FOLDER + filename, is_cropped=False, is_distorted=True)
    #     image = image.undistort()
    #     image = image.crop()
    #     image.saveToImagePath(OUTPUT_BASE_FOLDER + IR_FOLDER + filename)
    #     fileroot = filename.split('.')[0]
    #     for partition_image in image.partitions():
    #         partition_filename = f"{fileroot}_p{partition_image.partition_coordinates[0]}{partition_image.partition_coordinates[1]}.JPG"
    #         partition_image.saveToImagePath(PARTITION_BASE_FOLDER + IR_FOLDER + partition_filename)
    #     print(f"Processed IR: {fileroot}")

    for filename in os.listdir(INPUT_BASE_FOLDER + LABEL_FOLDER):
        label_set = LabelSet.loadFromFilePath(INPUT_BASE_FOLDER + LABEL_FOLDER + filename, is_cropped=False)
        # label_set = label_set.crop()
        # label_set.writeToFilePath(CROPPED_BASE_FOLDER + LABEL_FOLDER + filename)
        fileroot = filename.split('.')[0]
        for partition_label_set in label_set.partitions():
            partition_filename = f"{fileroot}_p{partition_label_set.partition_coordinates[0]}{partition_label_set.partition_coordinates[1]}.txt"
            partition_label_set.writeToFilePath(PARTITION_BASE_FOLDER + LABEL_FOLDER + partition_filename)
        print(f"Processed labelfile: {fileroot}")


if __name__ == "__main__":
    transform_data()
    show_blended_output("2021_09_holtan_0535", partition=(2,1))

        