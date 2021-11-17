import os
import shutil
from helpers import GET_VALIDATION_SET_FILEROOTS
    
# base folders
TRAIN_BASE_FOLDER = '../data/train/'
VALIDATION_BASE_FOLDER = '../data/validation/'

# base folder structure
RGB_FOLDER = 'rgb/'
IR_FOLDER = 'ir/'
LABEL_FOLDER = 'labels/'
    

validation_set_fileroots = GET_VALIDATION_SET_FILEROOTS() # Partition: (4,3) , Cropped: (3,2)

# Move validation images and labels
for fileroot in validation_set_fileroots:
    try:
        filename = fileroot + '.JPG'
        print(TRAIN_BASE_FOLDER + IR_FOLDER + filename, VALIDATION_BASE_FOLDER + IR_FOLDER + filename)
        shutil.move(TRAIN_BASE_FOLDER + IR_FOLDER + filename, VALIDATION_BASE_FOLDER + IR_FOLDER)
    except:
        continue
    # filename = fileroot + '.txt'
    # print(PARTITION_BASE_FOLDER + LABEL_FOLDER + filename, VALIDATION_BASE_FOLDER + LABEL_FOLDER + filename)
    # shutil.move(PARTITION_BASE_FOLDER + LABEL_FOLDER + filename, VALIDATION_BASE_FOLDER + LABEL_FOLDER)


# for filename in os.listdir(PARTITION_BASE_FOLDER + RGB_FOLDER):
#     os.rename(PARTITION_BASE_FOLDER + RGB_FOLDER + filename, TRAIN_BASE_FOLDER + RGB_FOLDER + filename)

# for filename in os.listdir(PARTITION_BASE_FOLDER + LABEL_FOLDER):
#     os.rename(PARTITION_BASE_FOLDER + LABEL_FOLDER + filename, TRAIN_BASE_FOLDER + LABEL_FOLDER + filename)


