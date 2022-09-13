# master-sau
This is a repository for the altered YOLOv5 codebase and scripts used in the Master Thesis - Improved Sheep Detection, written in the spring of 2022.
The altered YOLOv5 model found in this repository makes it possible to run YOLOv5 with a fourth channel of information, in our case this was Infrared Radiation (IR) imagery. 

The branch master only contains the standard model to process RGB images.

The branch four-input contains the model which combines RGB images and IR images as 4 channels before processing the image.

The branch generic-fusion contains the model which processes RGB images and IR images separetly in the backbone, and combines the results in a fusion module before the neck/head. 


The data folder is ignored, but should look like this:

- data/
  - train/
    - images/*.jpg
    - ir/*.jpg
    - labels/*.txt
  - validation/
    - images/*.jpg
    - ir/*.jpg
    - labels/*.txt
  - test/
    - images/*.jpg
    - ir/*.jpg
    - labels/*.txt
