from typing import List
from Label import Label
from helpers import GET_PARTITION_TOP_LEFT_CORNER

class LabelSet():
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
            return LabelSet(labels, is_cropped, partition_coordinates)

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
                    new_label.confidence = new_label.area() * new_label.confidence # TODO: Remove this!
                    new_labels.append(new_label)

        new_label_set = LabelSet(new_labels, is_cropped)
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

        print("beforeNonMax:", len(self.labels))
        print("afterNonMax:", len(kept_labels))
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


if __name__ == "__main__":
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