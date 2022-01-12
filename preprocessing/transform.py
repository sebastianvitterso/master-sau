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
from models import LabelSet, Image

# base folders
INPUT_BASE_FOLDER = '../data/train/'
CROPPED_BASE_FOLDER = '../data-cropped/train/'
PARTITION_BASE_FOLDER = '../data-partitioned/train/'
CROPPED_PARTITION_BASE_FOLDER = '../data-cropped-partitioned/train/'

# base folder structure
RGB_FOLDER = 'images/'
IR_FOLDER = 'ir/'
LABEL_FOLDER = 'labels/'

def show_blended_output(fileroot:str, partition_coordinates:'tuple[int, int]'=None, use_ir:bool=False, save:bool=False):
    '''@param filename The filename without the filetype/ending. E.g. `2021_09_holtan_0535`'''

    log_string = f"Showing image {fileroot}, {'with' if use_ir else 'without'} IR."
    if(partition_coordinates is not None):
        log_string += f" Partition: {partition_coordinates}."
    print(log_string)

    base_folder = CROPPED_BASE_FOLDER if use_ir else INPUT_BASE_FOLDER
    if(partition_coordinates is not None):
        base_folder = PARTITION_BASE_FOLDER
        fileroot = f"{fileroot}_p{partition_coordinates[0]}{partition_coordinates[1]}"

    rgb_image = Image.loadFromImagePath(base_folder + RGB_FOLDER + fileroot + ".JPG", is_cropped=use_ir, partition_coordinates=partition_coordinates)
    if(use_ir):
        ir_image = Image.loadFromImagePath(base_folder + IR_FOLDER + fileroot + ".JPG", is_cropped=use_ir, partition_coordinates=partition_coordinates)
    label_set = LabelSet.loadFromFilePath(base_folder + LABEL_FOLDER + fileroot + ".txt", is_cropped=use_ir, partition_coordinates=partition_coordinates)

    plt.figure()

    if(use_ir):
        blended_img = cv2.cvtColor(np.maximum(rgb_image.img, ir_image.img), cv2.COLOR_BGR2RGB)
        for label in label_set.labels:
            blended_img = cv2.rectangle(blended_img, (label.left, label.top), (label.right, label.bottom), (0,0,255), 2)
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(rgb_image.img, cv2.COLOR_BGR2RGB))
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(ir_image.img, cv2.COLOR_BGR2RGB))
        plt.subplot(2, 3, (2,6))
        plt.imshow(blended_img)
    else:
        img = cv2.cvtColor(rgb_image.img, cv2.COLOR_BGR2RGB)
        for label in label_set.labels:
            img = cv2.rectangle(img, (label.left, label.top), (label.right, label.bottom), (0,0,255), 2)
        plt.imshow(img)

    # plt.get_current_fig_manager().full_screen_toggle()
    plt.title(fileroot)
    plt.show()


def transform_data(use_ir:bool=True, partition:bool=True, keep_empty:bool=True):
    assert use_ir or partition, "Why even run this thing, if you're not gonna use_ir or partition..?"

    fileroots = set() # Look-up for fileroots we want to save. Not relevent if `keep_empty=True`

    filenames = os.listdir(INPUT_BASE_FOLDER + LABEL_FOLDER)
    filenames.sort()
    for filename in filenames:
        fileroot = filename.split('.')[0]
        label_set = LabelSet.loadFromFilePath(INPUT_BASE_FOLDER + LABEL_FOLDER + filename, is_cropped=False)
        if(use_ir):
            label_set = label_set.crop()
            if (len(label_set.labels) > 0 or keep_empty) and not partition:
                fileroots.add(fileroot)
                label_set.writeToFilePath(CROPPED_BASE_FOLDER + LABEL_FOLDER + fileroot + '.txt')

        if(partition):
            for partition_label_set in label_set.partitions():
                if len(partition_label_set.labels) > 0 or keep_empty:
                    x, y = partition_label_set.partition_coordinates
                    partition_fileroot = f"{fileroot}_p{x}{y}"
                    fileroots.add(partition_fileroot)
                    base_folder = CROPPED_PARTITION_BASE_FOLDER if use_ir else PARTITION_BASE_FOLDER
                    partition_label_set.writeToFilePath(base_folder + LABEL_FOLDER + partition_fileroot + '.txt')

        print(f"Processed labelfile: {fileroot}")

    filenames = os.listdir(INPUT_BASE_FOLDER + RGB_FOLDER)
    filenames.sort()
    for filename in filenames:
        fileroot = filename.split('.')[0]
        image = Image.loadFromImagePath(INPUT_BASE_FOLDER + RGB_FOLDER + filename, is_cropped=False)
        if(use_ir):
            image = image.crop()
            if(fileroot in fileroots or keep_empty):
                image.saveToImagePath(CROPPED_BASE_FOLDER + RGB_FOLDER + filename)

        if(partition):
            for partition_image in image.partitions():
                x, y = partition_image.partition_coordinates
                partition_fileroot = f"{fileroot}_p{x}{y}"
                if(partition_fileroot in fileroots or keep_empty):
                    base_folder = CROPPED_PARTITION_BASE_FOLDER if use_ir else PARTITION_BASE_FOLDER
                    partition_image.saveToImagePath(base_folder + RGB_FOLDER + partition_fileroot + '.JPG')

        print(f"Processed RGB: {fileroot}")


    if(use_ir):
        filenames = os.listdir(INPUT_BASE_FOLDER + IR_FOLDER)
        filenames.sort()
        for filename in filenames:
            fileroot = filename.split('.')[0]
            image = Image.loadFromImagePath(INPUT_BASE_FOLDER + IR_FOLDER + filename, is_cropped=False, is_distorted=True)
            image = image.undistort()
            image = image.crop()
            if(fileroot in fileroots or keep_empty):
                image.saveToImagePath(CROPPED_BASE_FOLDER + IR_FOLDER + filename)

            if(partition):
                for partition_image in image.partitions():
                    x, y = partition_image.partition_coordinates
                    partition_fileroot = f"{fileroot}_p{x}{y}"
                    if(partition_fileroot in fileroots or keep_empty):
                        partition_image.saveToImagePath(CROPPED_PARTITION_BASE_FOLDER + IR_FOLDER + partition_fileroot + '.JPG')

            print(f"Processed IR: {fileroot}")



if __name__ == "__main__":
    transform_data(use_ir=False, partition=True, keep_empty=True)
    # show_blended_output("2020_05_orkanger_0960", partition_coordinates=None, use_ir=True)

        