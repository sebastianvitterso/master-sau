import yolo
import cv2
import numpy as np
import torch
from torch import nn, F

# yolov5s = yolo.Model()

# rgb_image = cv2.imread('../../raw_data/kari/RGB/2019_08_storli1_0702.jpg')
# print('RGB shape: ', rgb_image.shape)

# rgb_image = np.array(rgb_image)
# rgb_image = torch.from_numpy(rgb_image)

# results = yolov5s(rgb_image, augment=False)

# results.print()  
# results.show()

class YoloFusionModel(nn.Module):
    def __init__(self):
        super(YoloFusionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
