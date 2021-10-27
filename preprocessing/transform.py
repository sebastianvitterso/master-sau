"""
Process for transforming raw images (RGB and IR), as well as (YOLO-)labels 
corresponding to the raw RGB images, into images (still RGB and IR) of the same size,
and making the labels fit the new (slightly cropped) images.

Depends on research done by Kari Meling Johannesen in her masters: https://github.com/thaiKari/sheepDetector
"""
import cv2
import os
from Label import Label
import transformation_helpers as helpers

INPUT_BASE_FOLDER = './data/input/'
OUTPUT_BASE_FOLDER = './data/output/'

RGB_FOLDER = 'rgb/'
IR_FOLDER = 'ir/'
LABEL_FOLDER = 'labels/'

def main():

    for rgb_filename in os.listdir(INPUT_BASE_FOLDER + RGB_FOLDER):
        raw_rgb = cv2.imread(INPUT_BASE_FOLDER + RGB_FOLDER + rgb_filename)
        cropped_rgb = helpers.crop_img(raw_rgb)
        cv2.imwrite(OUTPUT_BASE_FOLDER + RGB_FOLDER + rgb_filename, cropped_rgb)
        print(f"Processed RGB: {rgb_filename}")

    for ir_filename in os.listdir(INPUT_BASE_FOLDER + IR_FOLDER):
        raw_ir = cv2.imread(INPUT_BASE_FOLDER + IR_FOLDER + ir_filename)
        transformed_ir = helpers.transform_IR_im_to_vis_coordinate_system(raw_ir)
        cropped_img = helpers.crop_img(transformed_ir)
        cv2.imwrite(OUTPUT_BASE_FOLDER + IR_FOLDER + ir_filename, cropped_img)
        print(f"Processed IR: {ir_filename}")

    for label_filename in os.listdir(INPUT_BASE_FOLDER + LABEL_FOLDER):
        raw_labels = helpers.read_label_file_lines(INPUT_BASE_FOLDER + LABEL_FOLDER + label_filename)
        new_labels = []
        for raw_label in raw_labels:
            new_label: Label = Label.fromLabelLine(raw_label, False)
            new_label.crop()
            if(new_label.area() > 0):
                new_labels.append(new_label)

        helpers.write_label_file(OUTPUT_BASE_FOLDER + LABEL_FOLDER + label_filename, new_labels)
        print(f"Processed labelfile: {label_filename}")
        
if(__name__ == "__main__"):
    main()
        
