import os
import shutil
from helpers import GET_VALIDATION_SET_FILEROOTS
    
# base folders
TRAIN_BASE_FOLDER = '../../data-cropped-partitioned-no-msx-test/train/'
VALIDATION_BASE_FOLDER = '../../data-cropped-partitioned-no-msx-test/test/'

# base folder structure
RGB_FOLDER = 'images/'
IR_FOLDER = 'ir/'
LABEL_FOLDER = 'labels/'
    

validation_set_fileroots = GET_VALIDATION_SET_FILEROOTS((3,2)) # Partition: (4,3) , Cropped: (3,2)
moved_rgb = 0
moved_ir = 0
moved_label = 0
# Move validation images and labels
for i, fileroot in enumerate(validation_set_fileroots):

    filename = fileroot + '.JPG'
    if os.path.exists(TRAIN_BASE_FOLDER + RGB_FOLDER + filename) and os.path.exists(VALIDATION_BASE_FOLDER + RGB_FOLDER):
        # print(TRAIN_BASE_FOLDER + RGB_FOLDER + filename, VALIDATION_BASE_FOLDER + RGB_FOLDER + filename)
        shutil.move(TRAIN_BASE_FOLDER + RGB_FOLDER + filename, VALIDATION_BASE_FOLDER + RGB_FOLDER)
        moved_rgb += 1
    
    if os.path.exists(TRAIN_BASE_FOLDER + IR_FOLDER + filename) and os.path.exists(VALIDATION_BASE_FOLDER + IR_FOLDER):
        # print(TRAIN_BASE_FOLDER + IR_FOLDER + filename, VALIDATION_BASE_FOLDER + IR_FOLDER + filename)
        shutil.move(TRAIN_BASE_FOLDER + IR_FOLDER + filename, VALIDATION_BASE_FOLDER + IR_FOLDER)
        moved_ir += 1
    
    filename = fileroot + '.txt'
    if os.path.exists(TRAIN_BASE_FOLDER + LABEL_FOLDER + filename) and os.path.exists(VALIDATION_BASE_FOLDER + LABEL_FOLDER):
        # print(TRAIN_BASE_FOLDER + LABEL_FOLDER + filename, VALIDATION_BASE_FOLDER + LABEL_FOLDER + filename)
        shutil.move(TRAIN_BASE_FOLDER + LABEL_FOLDER + filename, VALIDATION_BASE_FOLDER + LABEL_FOLDER)
        moved_label += 1
    
    print(f'RGB: {moved_rgb} | IR: {moved_ir} | LABEL: {moved_label} | {i + 1} / {len(validation_set_fileroots)}')


