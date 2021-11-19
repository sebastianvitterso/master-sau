from typing import List
from Label import Label

class LabelSet():
    def __init__(self, labels:List['Label'], is_cropped:bool=False, partition_coordinates:'tuple[int, int]'=None):
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
            return LabelSet(labels, is_cropped, partition_coordinates)

    def writeToFilePath(self, file_path:str):
        label_lines = list(map(lambda label:label.toLabelLine(), self.labels))
        label_file_text = '\n'.join(label_lines)
        with open(file_path, 'w') as file:
            file.write(label_file_text)

    def crop(self):
        assert not self.is_cropped, "Don't crop a cropped label-set!"
        assert self.partition_coordinates is None, "Don't crop a partition label-set!"

        labels = []
        for raw_label in self.labels:
            cropped_label = raw_label.crop()
            if(cropped_label.area() > 0):
                labels.append(cropped_label)

        return LabelSet(labels, is_cropped=True)

    def partitions(self) -> List['LabelSet']:
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

