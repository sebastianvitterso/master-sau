"""
@source: https://github.com/thaiKari/sheepDetector/blob/master/transformations.py
"""

# coordinates are (x, y), measured in pixels, respectively from left and top.
# cropped coordinates were retrieved simply by measuring in photopea.com
CORNER_TOP_LEFT = (480, 285)
CORNER_BOTTOM_RIGHT = (3680, 2608)
RAW_SIZE_RGB = (4056, 3040)
RAW_SIZE_IR = (640, 480)
PARTITION_SIZE = (1280, 1280)
CROPPED_SIZE = (
    CORNER_BOTTOM_RIGHT[0] - CORNER_TOP_LEFT[0],
    CORNER_BOTTOM_RIGHT[1] - CORNER_TOP_LEFT[1],
)

def GET_PARTITION_TOP_LEFT_CORNER(x_coord, y_coord, is_cropped=False) -> tuple[int, int]:
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
