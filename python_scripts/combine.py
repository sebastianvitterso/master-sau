"""
Script to combine results from partitioned runs, to form
single predictions for single images (not 6 or 12 results
per image.)
"""

from collections import defaultdict
import os
from typing import List
from models import LabelSet

PREDICTION_BASE_FOLDER = '../../predictions/'
COMBINED_PREDICTION_BASE_FOLDER = '../../predictions_combined/'
GROUND_TRUTH_PARTITION_FOLDER = '../../data-partitioned/train/labels/'
GROUND_TRUTH_FOLDER = '../../data/train/labels/'

def combineResults(folder:str, is_cropped:bool=False, save:bool=False):
    prediction_folder = PREDICTION_BASE_FOLDER + folder
    prediction_filenames = os.listdir(prediction_folder)

    prediction_roots = []
    prediction_sets = defaultdict(lambda: [])

    combined_sets = []

    for prediction_filename in prediction_filenames:
        px,py = int(prediction_filename[-6:-5]), int(prediction_filename[-5:-4])
        label_set = LabelSet.loadFromFilePath(prediction_folder + prediction_filename, False, ( px, py ) )
        label_set.removeLowConfidenceLabels(0.5)

        # if prediction name is 2019_08_storli1_0740_p11.txt, then the root is 2019_08_storli1_0740 (which is 8 characters shorter)
        prediction_root = prediction_filename[:-8]
        prediction_roots.append(prediction_root)
        prediction_sets[prediction_root].append(label_set)


    for prediction_root, prediction_label_sets in prediction_sets.items():
        
        x_count, y_count = (3,2) if is_cropped else (4,3)
        
        partition_label_sets:'List[List[LabelSet|None]]' = []
        for y in range(y_count):
            partition_label_sets.append([])
            for x in range(x_count):
                partition_label_set:'LabelSet|None' = [*[label_set for label_set in prediction_label_sets if label_set.partition_coordinates == (x,y)], None][0]
                partition_label_sets[-1].append(partition_label_set)

        combined_label_set = LabelSet.fromPartitions(partition_label_sets)
        combined_sets.append(combined_label_set)
        
        if(save and len(combined_label_set.labels) > 0): 
            combined_label_set.writeToFilePath(COMBINED_PREDICTION_BASE_FOLDER + folder + prediction_root + '.txt', with_confidence=True)
    
    return combined_sets



if __name__ == "__main__":
    combineResults('partitioned_rgb_01/', save=True)