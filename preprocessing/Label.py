from helpers import RAW_SIZE_RGB, CROPPED_SIZE, PARTITION_SIZE, CORNER_TOP_LEFT, GET_PARTITION_TOP_LEFT_CORNER

class Label():
    def __init__(self, top:int, bottom:int, left:int, right:int, category:int, is_cropped:bool, is_partition:bool=False):
        self.top = top
        self.bottom = bottom
        self.right = right
        self.left = left
        self.category = category
        self.is_cropped = is_cropped
        self.is_partition = is_partition
        
    def __str__(self):
        return f"<Label is_cropped={self.is_cropped}, is_cropped={self.is_partition}, category={self.category}, top={self.top}, bottom={self.bottom}, right={self.right}, left={self.left}>"
    def __repr__(self):
        return self.__str__()

    @classmethod
    def fromLabelLine(cls, label_line:str, is_cropped:bool, is_partition:bool=False):
        ''' label_line should look something like `0 0.546844181459566 0.53125 0.008382642998027613 0.013157894736842105` '''
        tokens = label_line.split(' ')
        category, center_x_relative, center_y_relative, width_relative, height_relative = int(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])

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

        return cls(top, bottom, left, right, category, is_cropped)

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
