from typing import List
from helpers import RAW_SIZE_RGB, CROPPED_SIZE, PARTITION_SIZE, CORNER_TOP_LEFT, GET_PARTITION_TOP_LEFT_CORNER

class Label():
    def __init__(self, top:int, bottom:int, left:int, right:int, category:int, is_cropped:bool):
        self.top = top
        self.bottom = bottom
        self.right = right
        self.left = left
        self.category = category
        self.is_cropped = is_cropped
        
    def __str__(self):
        return f"<Label is_cropped={self.is_cropped}, category={self.category}, top={self.top}, bottom={self.bottom}, right={self.right}, left={self.left}>"
    def __repr__(self):
        return self.__str__()

    @classmethod
    def fromLabelLine(cls, label_line:str, is_cropped:bool):
        ''' label_line should look something like `0 0.546844181459566 0.53125 0.008382642998027613 0.013157894736842105` '''
        tokens = label_line.split(' ')
        category, center_x_relative, center_y_relative, width_relative, height_relative = int(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])

        SIZE = CROPPED_SIZE if is_cropped else RAW_SIZE_RGB
        # Explanation inside and outward:
        # 1. Transform from center-position to edge-position:   a = (center_x_relative - (width_relative  / 2))
        # 2. Transform from relative to pixel-position:         b = SIZE[0] * a
        # 3. Transform from float to int:                       c = int(b)
        # 4. Make sure the value doesn't go outside the SIZE:   d = max(c, 0)
        left =   max(int(SIZE[0] * (center_x_relative - (width_relative  / 2))), 0)
        right =  min(int(SIZE[0] * (center_x_relative + (width_relative  / 2))), SIZE[0] - 1)
        top =    max(int(SIZE[1] * (center_y_relative - (height_relative / 2))), 0)
        bottom = min(int(SIZE[1] * (center_y_relative + (height_relative / 2))), SIZE[1] - 1)

        return cls(top, bottom, left, right, category, is_cropped)

    def toLabelLine(self):
        SIZE = CROPPED_SIZE if self.is_cropped else RAW_SIZE_RGB
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

    def crop(self, TOP_LEFT_OFFSET:tuple[int, int], CROP_SIZE:tuple[int, int]):
        top =       min(max(self.top    - TOP_LEFT_OFFSET[1], 0), CROP_SIZE[1]-1)
        bottom =    min(max(self.bottom - TOP_LEFT_OFFSET[1], 0), CROP_SIZE[1]-1)
        left =      min(max(self.left   - TOP_LEFT_OFFSET[0], 0), CROP_SIZE[0]-1)
        right =     min(max(self.right  - TOP_LEFT_OFFSET[0], 0), CROP_SIZE[0]-1)

        return Label(top, bottom, left, right, self.category, True)
    
    def area(self):
        height = self.bottom - self.top
        width = self.right - self.left
        return height*width



class LabelSet():
    def __init__(self, labels:List['Label'], is_cropped:bool=False, partition_coordinates:tuple[int, int]=None):
        self.labels = labels
        self.is_cropped = is_cropped
        self.partition_coordinates = partition_coordinates
        
    def __str__(self):
        return f"<LabelSet len(labels)={len(self.labels)}, is_cropped={self.is_cropped}, partition_coordinates={self.partition_coordinates}>"
    def __repr__(self):
        return self.__str__()

    @classmethod
    def loadFromFilePath(cls, file_path:str, is_cropped:bool) -> 'LabelSet':
        with open(file_path) as file:
            labels = list(map(lambda line:Label.fromLabelLine(line.strip(), is_cropped), file.readlines()))
            return LabelSet(labels, is_cropped)

    def writeToFilePath(self, file_path:str):
        label_lines = list(map(lambda label:label.toLabelLine(), self.labels))
        label_file_text = '\n'.join(label_lines)
        with open(file_path, 'w') as file:
            file.write(label_file_text)

    def crop(self, TOP_LEFT_OFFSET:tuple[int, int]=CORNER_TOP_LEFT, CROP_SIZE:tuple[int, int]=CROPPED_SIZE):
        # CORNER_TOP_LEFT, CROPPED_SIZE
        assert self.partition_coordinates is None, "Don't crop a partition label-set!"

        labels = []
        for raw_label in self.labels:
            cropped_label = raw_label.crop(TOP_LEFT_OFFSET, CROP_SIZE)
            if(cropped_label.area() > 0):
                labels.append(cropped_label)

        return LabelSet(labels, is_cropped=True)

    def partitions(self) -> List['LabelSet']:
        assert self.partition_coordinates is None, "Don't partition a partition!"

        x_partition_count = 3 if self.is_cropped else 4
        y_partition_count = 2 if self.is_cropped else 3

        partitions = []
        for y in range(y_partition_count):
            for x in range(x_partition_count):
                partition_top_left = GET_PARTITION_TOP_LEFT_CORNER(x, y, is_cropped=self.is_cropped)
                partition_labels = self.crop(partition_top_left, PARTITION_SIZE).labels
                partitions.append(LabelSet(partition_labels, is_cropped=self.is_cropped, partition_coordinates=(x,y)))

        return partitions
    

