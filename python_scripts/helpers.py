"""
@source: https://github.com/thaiKari/sheepDetector/blob/master/transformations.py
"""

# coordinates are (x, y), measured in pixels, respectively from left and top.
# cropped coordinates were retrieved simply by measuring in photopea.com
import os
from typing import List

CORNER_TOP_LEFT = (480, 285)
CORNER_BOTTOM_RIGHT = (3680, 2608)
RAW_SIZE_RGB = (4056, 3040)
RAW_SIZE_IR = (640, 480)
PARTITION_SIZE = (1280, 1280)
CROPPED_SIZE = ( # (3200, 2323)
    CORNER_BOTTOM_RIGHT[0] - CORNER_TOP_LEFT[0],
    CORNER_BOTTOM_RIGHT[1] - CORNER_TOP_LEFT[1],
)

def GET_PARTITION_TOP_LEFT_CORNER(x_coord, y_coord, is_cropped=False) -> 'tuple[int, int]':
    box = PARTITION_TOP_LEFT_CORNERS_CROPPED if is_cropped else PARTITION_TOP_LEFT_CORNERS_RAW
    return box[y_coord][x_coord]

PARTITION_TOP_LEFT_CORNERS_CROPPED = (
    ((0,   0), (960,   0), (1920,    0)),
    ((0,1043), (960,1043), (1920, 1043)),
)
PARTITION_TOP_LEFT_CORNERS_RAW = (
    ((0,   0), (925,   0), (1851,    0), (2776,    0)),
    ((0, 880), (925, 880), (1851,  880), (2776,  880)),
    ((0,1760), (925,1760), (1851, 1760), (2776, 1760)),
)


GRID_SIZE = (8,7)

# OLD VALIDATION SET
VALIDATION_SET = (
    ('2019_08_storli1_', (740, 782)),
    ('2019_08_storli1_', (1421, 1471)),
    ('2019_08_storli1_', (1545, 1569)),
    ('2019_08_storli1_', (3333, 3375)),
    ('2019_09_storli2_', (1270, 1316)),
    ('2019_09_storli2_', (2497, 2531)),
    ('2019_10_storli3_', (3134, 3174)),
    ('2019_10_storli3_', (3708, 3750)),
    ('2020_05_klabo_', (554, 580)),
    ('2020_05_orkanger_', (730, 742)),
    ('2021_09_holtan_', (539, 545)),
    ('2021_09_holtan_', (1074, 1092)),
    ('2021_09_holtan_', (1150, 1192)),
    ('2021_09_holtan_', (1208, 1232)),
    ('2021_09_holtan_', (1750, 1794)),
    ('2021_09_holtan_', (2201, 2223)),
    ('2021_10_holtan_', (2257, 2283)),
    ('2021_10_holtan_', (2521, 2589)),
)

VALIDATION_BASE_FOLDER = '../../data-cropped-no-msx-test/validation/images/'

def GET_VALIDATION_SET_FILEROOTS(partitioning_size:'tuple[int,int]'=None) -> List[str]:
    validation_set = []

    partitioning_set = ['']
    if partitioning_size is not None:
        partitioning_set = []
        for x in range(partitioning_size[0]):
            for y in range(partitioning_size[1]):
                partitioning_set.append(f'_p{x}{y}')

    filenames = os.listdir(VALIDATION_BASE_FOLDER)
    filenames.sort()
    for filename in filenames:
        fileroot = filename.split('.')[0]
        for partition in partitioning_set:
            partition_name = fileroot + partition
            validation_set.append(partition_name)

    # for prefix, (from_num, to_num) in VALIDATION_SET:
    #     for i in range(from_num, to_num + 1):
    #         name = f'{prefix}{i:04}'
    #         for partition in partitioning_set:
    #             partition_name = name + partition
    #             validation_set.append(partition_name)
    
    return validation_set