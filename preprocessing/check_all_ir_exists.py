import os
import shutil
from helpers import GET_VALIDATION_SET_FILEROOTS
    
# base folders
BASE_FOLDER = '../data-cropped/validation/'

# base folder structure
RGB_FOLDER = 'images/'
IR_FOLDER = 'ir/'
LABEL_FOLDER = 'labels/'

for filename in os.listdir(BASE_FOLDER + RGB_FOLDER):
    if not os.path.isfile(BASE_FOLDER + IR_FOLDER + filename):
        print(BASE_FOLDER + RGB_FOLDER + filename)



