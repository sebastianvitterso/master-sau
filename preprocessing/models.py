from typing import List
from skimage import transform
import numpy as np
import math
import cv2

from helpers import RAW_SIZE_RGB, RAW_SIZE_IR, CROPPED_SIZE, PARTITION_SIZE, CORNER_TOP_LEFT, CORNER_BOTTOM_RIGHT, GET_PARTITION_TOP_LEFT_CORNER






 #                                    #####               
 #         ##   #####  ###### #      #     # ###### ##### 
 #        #  #  #    # #      #      #       #        #   
 #       #    # #####  #####  #       #####  #####    #   
 #       ###### #    # #      #            # #        #   
 #       #    # #    # #      #      #     # #        #   
 ####### #    # #####  ###### ######  #####  ######   #   
                                                          
class LabelSet():
    label_confidence_threshold = 0.5

    def __init__(self, labels:'List[Label]', is_cropped:bool=False, partition_coordinates:'tuple[int, int]'=None):
        self.labels = labels
        self.is_cropped = is_cropped
        self.partition_coordinates = partition_coordinates
        
    def __str__(self):
        return f"<LabelSet len(labels)={len(self.labels)}, is_cropped={self.is_cropped}, partition_coordinates={self.partition_coordinates}>"
    def __repr__(self):
        return self.__str__()

    @classmethod
    def loadFromFilePath(cls, file_path:str, is_cropped:bool, partition_coordinates:'tuple[int, int]'=None) -> 'LabelSet':
        is_partition = partition_coordinates is not None
        with open(file_path) as file:
            labels = list(map(lambda line:Label.fromLabelLine(line.strip(), is_cropped, is_partition), file.readlines()))
            labels = list(filter(lambda label: label.confidence >= cls.label_confidence_threshold, labels)) # Only keep l+abels with confidence >= 0.3
            return cls(labels, is_cropped, partition_coordinates)

    def writeToFilePath(self, file_path:str):
        label_lines = list(map(lambda label:label.toLabelLine(), self.labels))
        label_file_text = '\n'.join(label_lines)
        with open(file_path, 'w') as file:
            file.write(label_file_text)

    @classmethod
    def fromPartitions(cls, label_sets:'List[List[LabelSet]]'):
        # if the first "row" has 3 entries, the image is cropped. Otherwise, it has 4 entries, and the image isn't cropped. 
        is_cropped = len(label_sets[0]) == 3  
        new_labels: 'List[Label]' = []

        for y, partition_label_set_row in enumerate(label_sets):
            for x, partition_label_set in enumerate(partition_label_set_row):
                offset_x, offset_y = GET_PARTITION_TOP_LEFT_CORNER(x, y, is_cropped)
                for label in partition_label_set.labels:
                    new_label = Label(
                        label.top + offset_y,
                        label.bottom + offset_y,
                        label.left + offset_x,
                        label.right + offset_x,
                        label.category,
                        is_cropped,
                        False,
                        label.confidence
                    )
                    new_labels.append(new_label)

        new_label_set = cls(new_labels, is_cropped)
        new_label_set.nonMaxSuppression()
        return new_label_set

    def nonMaxSuppression(self, threshold=0.0):
        remaining_labels = self.labels.copy()
        kept_labels:'List[Label]' = []
        while len(remaining_labels) > 0:
            max_confidence_index = 0
            max_confidence_value = -1
            for i, label in enumerate(remaining_labels):
                if(label.confidence > max_confidence_value):
                    max_confidence_value = label.confidence
                    max_confidence_index = i

            keep_label = remaining_labels.pop(max_confidence_index)
            kept_labels.append(keep_label)

            for label in remaining_labels.copy():
                if(keep_label.getIntersectionOverUnion(label) > threshold):
                    remaining_labels.remove(label)

        self.labels = kept_labels

    def crop(self):
        assert not self.is_cropped, "Don't crop a cropped label-set!"
        assert self.partition_coordinates is None, "Don't crop a partition label-set!"

        labels = []
        for raw_label in self.labels:
            cropped_label = raw_label.crop()
            if(cropped_label.area() > 0):
                labels.append(cropped_label)

        return LabelSet(labels, is_cropped=True)

    def partitions(self) -> 'List[LabelSet]':
        assert self.partition_coordinates is None, "Don't partition a partition label-set!"

        x_partition_count = 3 if self.is_cropped else 4
        y_partition_count = 2 if self.is_cropped else 3

        partitions = []
        for y in range(y_partition_count):
            for x in range(x_partition_count):
                partitions.append(self.partition((x,y)))

        return partitions

    def partition(self, partition_coordinates:'tuple[int, int]') -> 'LabelSet':
        assert self.partition_coordinates is None, "Don't partition a partition label-set!"

        partition_labels = list(map(lambda label: label.partition(partition_coordinates), self.labels))
        partition_labels = list(filter(lambda partition_label: partition_label.area() > 0, partition_labels))
        return LabelSet(partition_labels, self.is_cropped, partition_coordinates)

    def should_show(self, prediction_label_set:'LabelSet'):

        for ground_truth_label in self.labels:
            has_overlap = False
            for prediction_label in prediction_label_set.labels:

                if ground_truth_label.getIntersection(prediction_label) > 0.2 * ground_truth_label.area():
                    has_overlap = True

            if not has_overlap:
                return True

        for prediction_label in prediction_label_set.labels:
            has_overlap = False
            for ground_truth_label in self.labels:

                if prediction_label.getIntersection(ground_truth_label) > 0.2 * prediction_label.area():
                    has_overlap = True

            if not has_overlap:
                return True

        return False








 #                                   
 #         ##   #####  ###### #      
 #        #  #  #    # #      #      
 #       #    # #####  #####  #      
 #       ###### #    # #      #      
 #       #    # #    # #      #      
 ####### #    # #####  ###### ###### 
                                     
class Label():
    def __init__(self, top:int, bottom:int, left:int, right:int, category:int, is_cropped:bool, is_partition:bool=False, confidence:float=1):
        self.top = top
        self.bottom = bottom
        self.right = right
        self.left = left
        self.category = category
        self.is_cropped = is_cropped
        self.is_partition = is_partition
        self.confidence = confidence 
        
    def __str__(self):
        return f"<Label is_cropped={self.is_cropped}, is_cropped={self.is_partition}, category={self.category}, confidence={self.confidence}, top={self.top}, bottom={self.bottom}, right={self.right}, left={self.left}>"
    def __repr__(self):
        return self.__str__()

    @classmethod
    def fromLabelLine(cls, label_line:str, is_cropped:bool, is_partition:bool=False):
        ''' 
        label_line should look something like `0 0.546844181459566 0.53125 0.008382642998027613 0.013157894736842105 [0.12314151]`.
        
        The last token in brackets is the confidence, which is optional.
        '''
        tokens = label_line.split(' ')
        category, center_x_relative, center_y_relative, width_relative, height_relative = int(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])
        confidence = 1 if len(tokens) < 6 else float(tokens[5])

        SIZE = PARTITION_SIZE if is_partition else (CROPPED_SIZE if is_cropped else RAW_SIZE_RGB)
        # Explanation inside and outward:
        # 1. Transform from center-position to edge-position:   a = (center_x_relative - (width_relative  / 2))
        # 2. Transform from relative to pixel-position:         b = SIZE[0] * a
        # 3. Transform from float to int:                       c = int(b)
        # 4. Make sure the value doesn't go outside the SIZE:   d = max(c, 0)
        left =   max(int(SIZE[0] * (center_x_relative - (width_relative  / 2))), 0)
        right =  min(int(SIZE[0] * (center_x_relative + (width_relative  / 2))), SIZE[0] - 1)
        top =    max(int(SIZE[1] * (center_y_relative - (height_relative / 2))), 0)
        bottom = min(int(SIZE[1] * (center_y_relative + (height_relative / 2))), SIZE[1] - 1)

        return cls(top, bottom, left, right, category, is_cropped, confidence=confidence)

    def toLabelLine(self):
        SIZE = PARTITION_SIZE if self.is_partition else (CROPPED_SIZE if self.is_cropped else RAW_SIZE_RGB)
        center_x_px = (self.left + self.right) / 2
        center_y_px = (self.top + self.bottom) / 2
        center_x_relative = center_x_px / SIZE[0]
        center_y_relative = center_y_px / SIZE[1]

        width_px = (self.right - self.left)
        height_px = (self.bottom - self.top)
        width_relative = width_px / SIZE[0]
        height_relative = height_px / SIZE[1]

        assert center_x_relative + (width_relative / 2) <= 1
        assert center_x_relative - (width_relative / 2) >= 0
        assert center_y_relative + (height_relative / 2) <= 1
        assert center_y_relative - (height_relative / 2) >= 0

        return f"{self.category} {center_x_relative} {center_y_relative} {width_relative} {height_relative}"

    def crop(self):
        assert not self.is_cropped, "Don't crop a cropped label!"

        top =       min(max(self.top    - CORNER_TOP_LEFT[1], 0), CROPPED_SIZE[1]-1)
        bottom =    min(max(self.bottom - CORNER_TOP_LEFT[1], 0), CROPPED_SIZE[1]-1)
        left =      min(max(self.left   - CORNER_TOP_LEFT[0], 0), CROPPED_SIZE[0]-1)
        right =     min(max(self.right  - CORNER_TOP_LEFT[0], 0), CROPPED_SIZE[0]-1)

        return Label(top, bottom, left, right, self.category, True)

    def partition(self, partition_coordinates:'tuple[int, int]'):
        assert not self.is_partition, "Don't partition a partition label!"

        x, y = partition_coordinates
        partition_top_left = GET_PARTITION_TOP_LEFT_CORNER(x, y, is_cropped=self.is_cropped)

        top =       min(max(self.top    - partition_top_left[1], 0), PARTITION_SIZE[1]-1)
        bottom =    min(max(self.bottom - partition_top_left[1], 0), PARTITION_SIZE[1]-1)
        left =      min(max(self.left   - partition_top_left[0], 0), PARTITION_SIZE[0]-1)
        right =     min(max(self.right  - partition_top_left[0], 0), PARTITION_SIZE[0]-1)

        return Label(top, bottom, left, right, self.category, self.is_cropped, is_partition=True)
    
    def area(self):
        height = self.bottom - self.top
        width = self.right - self.left
        return height*width


    def getIntersection(self, label:'Label'):
        max_top = max(self.top, label.top)
        min_bottom = min(self.bottom, label.bottom)
        max_left = max(self.left, label.left)
        min_right = min(self.right, label.right)

        # if they don't overlap at all, return 0.
        if(min_bottom < max_top or min_right < max_left):
            return 0

        height = min_bottom - max_top
        width = min_right - max_left
        return height*width

    def getUnion(self, label:'Label'):
        return self.area() + label.area() - self.getIntersection(label)

    def getIntersectionOverUnion(self, label:'Label'):
        return self.getIntersection(label) / self.getUnion(label)





  #####                  #                                    #####               
 #     # #####  # #####  #         ##   #####  ###### #      #     # ###### ##### 
 #       #    # # #    # #        #  #  #    # #      #      #       #        #   
 #  #### #    # # #    # #       #    # #####  #####  #       #####  #####    #   
 #     # #####  # #    # #       ###### #    # #      #            # #        #   
 #     # #   #  # #    # #       #    # #    # #      #      #     # #        #   
  #####  #    # # #####  ####### #    # #####  ###### ######  #####  ######   #   
                                                                                  
class GridLabelSet():
    def __init__(self, is_cropped:bool, grid:np.ndarray):
        self.is_cropped = is_cropped
        self.grid = grid

    @classmethod
    def from_label_set(cls, label_set:LabelSet, is_prediction:bool=False):
        grid_label_constructor_vectorized = np.vectorize(lambda x,y: GridLabel.from_coordinates(x,y, label_set.is_cropped))
        grid:np.ndarray = np.fromfunction(grid_label_constructor_vectorized, (8,7))
        
        for label in label_set.labels:
            for ix, iy in np.ndindex(grid.shape):
                grid_label:GridLabel = grid[ix, iy]

                # `is_prediction` is True if we're transforming a prediction LabelSet, 
                # and False if we're transforming a ground truth LabelSet.
                if(is_prediction): 
                    if grid_label.get_overlap_ratio(label) > 0.1:
                        grid_label.value = True
                        grid_label.sheep_count += 1
                else:
                    if grid_label.get_overlap_ratio(label) > 0.2:
                        grid_label.value = True
                        grid_label.sheep_count += 1
                    elif grid_label.get_overlap_ratio(label) > 0 and grid_label.value != True:
                        grid_label.value = None # None means it's uncertain, and should not be counted.

        return cls(label_set.is_cropped, grid)

    def compare(self, prediction_grid_label_set:'GridLabelSet'):
        """ Should be used like this: `ground_truth_grid_label_set.compare(prediction_grid_label_set)` """
        tp = 0 # true positive counter
        tn = 0 # true negative counter
        fp = 0 # false positive counter
        fn = 0 # false negative counter

        total_sheep_count = 0
        found_sheep_count = 0

        for ix, iy in np.ndindex(self.grid.shape):
            ground_truth_grid_label:'GridLabel' = self.grid[ix, iy]
            prediction_grid_label:'GridLabel' = prediction_grid_label_set.grid[ix, iy]

            if(ground_truth_grid_label.value == True and prediction_grid_label.value == True): tp += 1
            if(ground_truth_grid_label.value == True and prediction_grid_label.value == False): fn += 1
            if(ground_truth_grid_label.value == False and prediction_grid_label.value == True): fp += 1
            if(ground_truth_grid_label.value == False and prediction_grid_label.value == False): tn += 1

            if ground_truth_grid_label.value == True:
                total_sheep_count += ground_truth_grid_label.sheep_count
                if prediction_grid_label.value == True:
                    found_sheep_count += ground_truth_grid_label.sheep_count


        return (tp, tn, fp, fn, total_sheep_count, found_sheep_count)




  #####                  #                                   
 #     # #####  # #####  #         ##   #####  ###### #      
 #       #    # # #    # #        #  #  #    # #      #      
 #  #### #    # # #    # #       #    # #####  #####  #      
 #     # #####  # #    # #       ###### #    # #      #      
 #     # #   #  # #    # #       #    # #    # #      #      
  #####  #    # # #####  ####### #    # #####  ###### ###### 
                                                             
class GridLabel():
    def __init__(self, bounding_box:'tuple[tuple[int, int],tuple[int, int]]'):
        self.value = False # can be True|False|None
        self.bounding_box = bounding_box
        self.sheep_count = 0

    @classmethod
    def from_coordinates(cls, x:int, y:int, is_cropped:bool):
        grid_size = (8,7)
        image_size = CROPPED_SIZE if is_cropped else RAW_SIZE_RGB
        x_min = 0 if x == 0 else math.ceil(image_size[0] * (x / grid_size[0]))
        x_max = math.ceil(image_size[0] * ((x + 1) / grid_size[0]))

        y_min = 0 if y == 0 else math.ceil(image_size[1] * (y / grid_size[1]))
        y_max = math.ceil(image_size[1] * ((y + 1) / grid_size[1]))

        return cls( ((x_min, x_max), (y_min, y_max)) )

    def get_overlap_ratio(self, label:Label):
        # Make a spoofy Label for this GridLabel, to get the intersection area of the given Label and this GridLabel.
        # Then divide that intersection area by the total area of the given label.
        ((x_min, x_max), (y_min, y_max)) = self.bounding_box
        if label.area():
            return label.getIntersection(Label(y_min, y_max, x_min, x_max, -1, label.is_cropped)) / label.area() 
        return 0





 ###                             
  #  #    #   ##    ####  ###### 
  #  ##  ##  #  #  #    # #      
  #  # ## # #    # #      #####  
  #  #    # ###### #  ### #      
  #  #    # #    # #    # #      
 ### #    # #    #  ####  ###### 
 
CAMERA_MATRIX_K = np.load("./parameters/camera_matrix_K.npy")
CAMERA_DIST_COEFFS = np.load("./parameters/camera_dist_coeffs.npy")
TRANSFORM_VIS_TO_IR = np.load("./parameters/Transform_vis_to_IR.npy")
TRANSFORM_IR_TO_VIS = np.load("./parameters/Transform_IR_to_Vis.npy")

class Image:
    def __init__(self, img:np.ndarray, is_distorted:bool=False, is_cropped:bool=False, partition_coordinates:'tuple[int, int]'=None):
        self.img = img
        self.is_distorted = is_distorted
        self.is_cropped = is_cropped
        self.partition_coordinates = partition_coordinates
        
    def __str__(self):
        return f"<Image img.shape={self.img.shape}, is_distorted={self.is_distorted}, is_cropped={self.is_cropped}, partition_coordinates={self.partition_coordinates}>"
    def __repr__(self):
        return self.__str__()

    @classmethod
    def loadFromImagePath(cls, image_path:str, is_distorted:bool=False, is_cropped:bool=False, partition_coordinates:'tuple[int, int]'=None):
        img = cv2.imread(image_path)
        return cls(img, is_distorted, is_cropped, partition_coordinates)

    def saveToImagePath(self, image_path:str):
        cv2.imwrite(image_path, self.img)

    def undistort(self) -> 'Image':
        ''' @source: https://github.com/thaiKari/sheepDetector/blob/master/transformations.py '''

        assert self.is_distorted, "Don't undistort an already undistorted image!"

        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX_K, CAMERA_DIST_COEFFS, RAW_SIZE_IR, 1, RAW_SIZE_IR)    
        img = cv2.undistort(self.img, CAMERA_MATRIX_K, CAMERA_DIST_COEFFS, None, newcameramtx)
        T = transform.AffineTransform(TRANSFORM_VIS_TO_IR)
        img = (transform.warp(img, T, output_shape=(RAW_SIZE_RGB[1], RAW_SIZE_RGB[0])) * 255).astype(int)
        return Image(img, is_distorted=False, is_cropped=self.is_cropped, partition_coordinates=self.partition_coordinates)

    def crop(self) -> 'Image':
        assert not self.is_cropped, "Don't crop an already cropped image!"
        img = self.img[ 
            CORNER_TOP_LEFT[1]:CORNER_BOTTOM_RIGHT[1],
            CORNER_TOP_LEFT[0]:CORNER_BOTTOM_RIGHT[0],
        ]
        return Image(img, is_distorted=self.is_distorted, is_cropped=True, partition_coordinates=self.partition_coordinates)


    def partitions(self) -> List['Image']:
        assert self.partition_coordinates is None, "Don't partition a partition!"

        x_partition_count = 3 if self.is_cropped else 4
        y_partition_count = 2 if self.is_cropped else 3

        partitions = []
        for y in range(y_partition_count):
            for x in range(x_partition_count):
                partition_top_left = GET_PARTITION_TOP_LEFT_CORNER(x, y, is_cropped=self.is_cropped)
                partition_img = self.img[
                    partition_top_left[1]:partition_top_left[1] + 1280,
                    partition_top_left[0]:partition_top_left[0] + 1280,
                ]
                partition_image = Image(partition_img, is_distorted=self.is_distorted, is_cropped=self.is_cropped, partition_coordinates=(x,y))
                partitions.append(partition_image)

        return partitions









 #######                                     
    #    ######  ####  ##### # #    #  ####  
    #    #      #        #   # ##   # #    # 
    #    #####   ####    #   # # #  # #      
    #    #           #   #   # #  # # #  ### 
    #    #      #    #   #   # #   ## #    # 
    #    ######  ####    #   # #    #  ####  
                                             
def testLabelsetFromPartitionsFunction():
    labelSet = LabelSet.loadFromFilePath('../data-cropped/labels/2020_05_orkanger_0732.txt', True)

    print("original:", len(labelSet.labels))
    partitionLabelSets = labelSet.partitions()
    partitionlabelSetRows: 'List[List[LabelSet]]' = []

    for y in range(2):
        partitionlabelSetRows.append([])
        for x in range(3):
            for partitionLabelSet in partitionLabelSets:
                if partitionLabelSet.partition_coordinates[0] == x and partitionLabelSet.partition_coordinates[1] == y:
                    partitionlabelSetRows[y].append(partitionLabelSet)

    joinedLabelSet = LabelSet.fromPartitions(partitionlabelSetRows)
    print(joinedLabelSet)

if __name__ == "__main__":
    testLabelsetFromPartitionsFunction()

