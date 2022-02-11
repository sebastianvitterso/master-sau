
from collections import defaultdict
import os
import cv2
import numpy as np
from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt

from models import LabelSet, Image, GridLabelSet, GridLabel

# base folders
# INPUT_BASE_FOLDER = '../../data/validation/'
INPUT_BASE_FOLDER = '../../data-cropped-no-msx/validation/'
CROPPED_BASE_FOLDER = '../../data-cropped-no-msx/validation/'
PARTITION_BASE_FOLDER = '../../data-partitioned/validation/'
CROPPED_PARTITION_BASE_FOLDER = '../../data-cropped-partition/validation/'

# base folder structure
RGB_FOLDER = 'images/'
IR_FOLDER = 'ir/'
LABEL_FOLDER = 'labels/' 
LABEL_FOLDER_COLORED = 'colored_labels/' 
LABEL_FOLDER_OBSCURED = 'obscured_labels/' 
SELECTED_LABEL_FOLDER = LABEL_FOLDER

# probably found in a validation run
PREDICTION_FOLDER = 'rgb-no-msx'
PREDICTION_PATH = f'../yolov5/runs/val/{PREDICTION_FOLDER}/labels/'

LABEL_CATEGORIES_COLORED = defaultdict(lambda: 'sheep', { 0: 'black sheep', 1: 'brown sheep', 2: 'grey sheep', 3: 'white sheep' })
LABEL_CATEGORIES_OBSCURED = defaultdict(lambda: 'sheep', { 0: 'sheep', 1: 'partially covered', 2: 'partially obscured', 3: 'completely obscured' })
SELECTED_LABEL_CATEGORIES = LABEL_CATEGORIES_COLORED

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def get_grid_metrics(fileroot:str, partition_coordinates:'tuple[int, int]'=None, use_ir:bool=False, show_image:bool=False, show_print:bool=False, conf:float=0.0):
    if show_print:
        print(f"\nCalculating grid-metrics for {fileroot}")

    base_folder = CROPPED_BASE_FOLDER if use_ir else INPUT_BASE_FOLDER
    if(partition_coordinates is not None):
        base_folder = PARTITION_BASE_FOLDER
        fileroot = f"{fileroot}_p{partition_coordinates[0]}{partition_coordinates[1]}"

    ground_truth_label_set = LabelSet.loadFromFilePath(base_folder + SELECTED_LABEL_FOLDER + fileroot + ".txt", is_cropped=use_ir, partition_coordinates=partition_coordinates)
    ground_truth_grid_label_set = GridLabelSet.from_label_set(ground_truth_label_set)
    prediction_label_set = LabelSet.loadFromFilePath(PREDICTION_PATH + fileroot + ".txt", is_cropped=use_ir, partition_coordinates=partition_coordinates)
    prediction_label_set.removeLowConfidenceLabels(conf)
    prediction_grid_label_set = GridLabelSet.from_label_set(prediction_label_set, is_prediction=True)

    (tp, tn, fp, fn, total_sheep_count, found_sheep_count, ground_truths, confidences) = ground_truth_grid_label_set.compare(prediction_grid_label_set)

    if show_image and ground_truth_label_set.has_mismatch(prediction_label_set):
        labels = (ground_truth_label_set, prediction_label_set, ground_truth_grid_label_set, prediction_grid_label_set)
        display_image(fileroot, base_folder, labels, partition_coordinates=partition_coordinates, use_ir=use_ir)

    if show_print:
        print(f"GRID METRICS {fileroot}")
        print("true_positive_count:", tp)
        print("true_negative_count:", tn)
        print("false_positive_count:", fp)
        print("false_negative_count:", fn)
        print("total_sheep_count:", total_sheep_count)
        print("found_sheep_count:", found_sheep_count)

    return tp, tn, fp, fn, total_sheep_count, found_sheep_count, ground_truths, confidences



def get_metrics(fileroot:str, partition_coordinates:'tuple[int, int]'=None, use_ir:bool=False, show_image:bool=False, show_print:bool=False, conf:float=0.0):
    '''@param filename The filename without the filetype/ending. E.g. `2021_09_holtan_0535`'''

    if show_print:
        print(f"\nCalculating metrics for {fileroot}")

    base_folder = CROPPED_BASE_FOLDER if use_ir else INPUT_BASE_FOLDER
    if(partition_coordinates is not None):
        base_folder = PARTITION_BASE_FOLDER
        fileroot = f"{fileroot}_p{partition_coordinates[0]}{partition_coordinates[1]}"

    ground_truth_label_set = LabelSet.loadFromFilePath(base_folder + SELECTED_LABEL_FOLDER + fileroot + ".txt", is_cropped=use_ir, partition_coordinates=partition_coordinates)
    prediction_label_set = LabelSet.loadFromFilePath(PREDICTION_PATH + fileroot + ".txt", is_cropped=use_ir, partition_coordinates=partition_coordinates)
    prediction_label_set.removeLowConfidenceLabels(conf)

    (tp, fp, conf, cats) = ground_truth_label_set.compare(prediction_label_set)
    total_sheep_count = len(ground_truth_label_set.labels)


    # category metrics [requires categoried labels]
    ground_truth_category_counter = defaultdict(lambda: 0)
    prediction_category_counter = defaultdict(lambda: 0)
    for label in ground_truth_label_set.labels:
        ground_truth_category_counter[label.category] += 1
    for category in cats[np.where(tp == 1)]: # count the category where we hit
        prediction_category_counter[int(category)] += 1


    if show_image and ground_truth_label_set.has_mismatch(prediction_label_set):
        labels = (ground_truth_label_set, prediction_label_set, None, None)
        display_image(fileroot, base_folder, labels, partition_coordinates=partition_coordinates, use_ir=use_ir)

    if show_print:
        print(f"METRICS {fileroot}")
        print("true_positive_count:", sum(tp))
        print("false_postive_count:", sum(fp))


    return tp, fp, conf, ground_truth_category_counter, prediction_category_counter, total_sheep_count


def display_image(fileroot:str, base_folder:str, labels:'tuple[LabelSet, LabelSet, GridLabelSet|None, GridLabelSet|None]', partition_coordinates:'tuple[int, int]'=None, use_ir:bool=False):
    log_string = f"Showing image {fileroot}, {'with' if use_ir else 'without'} IR."
    if(partition_coordinates is not None):
        log_string += f" Partition: {partition_coordinates}."
    print(log_string)

    rgb_image = Image.loadFromImagePath(base_folder + RGB_FOLDER + fileroot + ".JPG", is_cropped=use_ir, partition_coordinates=partition_coordinates)
    if(use_ir):
        ir_image = Image.loadFromImagePath(base_folder + IR_FOLDER + fileroot + ".JPG", is_cropped=use_ir, partition_coordinates=partition_coordinates)

    plt.figure()
    plt.title(fileroot)

    base_image = cv2.cvtColor(rgb_image.img, cv2.COLOR_BGR2RGB)

    if(use_ir):
        base_image = cv2.cvtColor(np.maximum(rgb_image.img, ir_image.img), cv2.COLOR_BGR2RGB)


    # GROUND TRUTH
    ground_truth_image = base_image.copy()
    (ground_truth_label_set, prediction_label_set, ground_truth_grid_label_set, prediction_grid_label_set) = labels

    if ground_truth_grid_label_set is not None:
        for ix, iy in np.ndindex(ground_truth_grid_label_set.grid.shape):
            grid_label:GridLabel = ground_truth_grid_label_set.grid[ix, iy]
            ((x_min, x_max), (y_min, y_max)) = grid_label.bounding_box
            bgr_color = (0,255,0) if grid_label.value == True else ((0,0,255) if grid_label.value == None else (150,150,150))
            ground_truth_image = cv2.rectangle(ground_truth_image, (x_min, y_min), (x_max, y_max), bgr_color, -1)

        ground_truth_image = cv2.addWeighted(base_image, 0.9, ground_truth_image, 0.1, 0)

    for label in ground_truth_label_set.labels:
        ground_truth_image = cv2.rectangle(ground_truth_image, (label.left, label.top), (label.right, label.bottom), (0,0,255), 2)

    plt.subplot(2, 2, 1) if use_ir else plt.subplot(1, 2, 1)
    plt.gca().set_title('Ground truth - ' + fileroot)
    plt.imshow(ground_truth_image)


    # PREDICTIONS
    prediction_image = base_image.copy()

    if ground_truth_grid_label_set is not None:
        for ix, iy in np.ndindex(prediction_grid_label_set.grid.shape):
            grid_label:GridLabel = prediction_grid_label_set.grid[ix, iy]
            ((x_min, x_max), (y_min, y_max)) = grid_label.bounding_box
            bgr_color = (0,255,0) if grid_label.value == True else ((255,0,0) if grid_label.value == None else (150,150,150))
            prediction_image = cv2.rectangle(prediction_image, (x_min, y_min), (x_max, y_max), bgr_color, -1)

        prediction_image = cv2.addWeighted(base_image, 0.9, prediction_image, 0.1, 0)

    for label in prediction_label_set.labels:
        prediction_image = cv2.rectangle(prediction_image, (label.left, label.top), (label.right, label.bottom), (255,0,0), 2)
        # prediction_image = cv2.putText(prediction_image, f'{label.confidence:.2}', (label.left, label.top - 12), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,0), 4, cv2.LINE_AA)

    plt.subplot(2, 2, 2) if use_ir else plt.subplot(1, 2, 2)
    plt.gca().set_title('Predictions')
    plt.imshow(prediction_image)

    if(use_ir):
        ir_image = cv2.cvtColor(ir_image.img, cv2.COLOR_BGR2RGB)
        for label in ground_truth_label_set.labels:
            ir_image = cv2.rectangle(ir_image, (label.left, label.top), (label.right, label.bottom), (0,0,255), 2)
        plt.subplot(2, 2, 3)
        plt.gca().set_title('IR')
        plt.imshow(ir_image)

    plt.get_current_fig_manager().full_screen_toggle()
    plt.tight_layout()
    plt.show()


def calculate_metrics(partition_coordinates:'tuple[int, int]'=None, use_ir:bool=False, show_image:bool=False, show_print:bool=False):
    print("\n====================")

    tp = np.array([])
    fp = np.array([])
    conf = np.array([])
    ground_truth_color_counter = defaultdict(lambda: 0)
    prediction_color_counter = defaultdict(lambda: 0)
    ground_truth_obscured_counter = defaultdict(lambda: 0)
    prediction_obscured_counter = defaultdict(lambda: 0)
    sheep_count = 0

    grid_tp = 0
    grid_tn = 0
    grid_fp = 0
    grid_fn = 0
    grid_total_sheep_count = 0
    grid_found_sheep_count = 0
    grid_ground_truths = np.array([])
    grid_confidences = np.array([])


    filenames = os.listdir(PREDICTION_PATH)
    filenames.sort()
    for filename in filenames:
        try:
            fileroot = filename.split('.')[0]
            file_tp, file_fp, file_conf, file_ground_truth_category_counter, file_prediction_category_counter, file_sheep_count = get_metrics(fileroot, partition_coordinates, use_ir, show_image, show_print)
            tp = np.concatenate([tp, file_tp])
            fp = np.concatenate([fp, file_fp])
            conf = np.concatenate([conf, file_conf])
            sheep_count += file_sheep_count

        except FileNotFoundError as e:
            print(e)
            # I had some missing labels... :/

    # Sort by confidence
    sort_order = np.argsort(-conf)
    tp, fp, conf = tp[sort_order], fp[sort_order], conf[sort_order]

    tpc = tp.cumsum(0)
    fpc = fp.cumsum(0)
    recall_curve = tpc / (sheep_count + 1e-16)  # recall curve
    precision_curve = tpc / (tpc + fpc + 1e-16)  # precision curve

    ap50s, mpre, mrec = compute_ap(recall_curve, precision_curve)

    px = np.linspace(0, 1, 1000)
    r = np.interp(-px, -conf, recall_curve, left=0) # negative x, xp because xp decreases
    p = np.interp(-px, -conf, precision_curve, left=1)  # p at pr_score
    f1 = 2 * p * r / (p + r + 1e-16)
    
    i = f1.argmax()  # max F1 index
    best_conf = i / 1000
    precision = p[i]
    recall = r[i]

    print(f"\nMETRICS - {PREDICTION_FOLDER}")
    print("AP@.5:", ap50s)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Confidence:", best_conf)
    
    global SELECTED_LABEL_FOLDER
    for filename in filenames:
        try:
            fileroot = filename.split('.')[0]

            SELECTED_LABEL_FOLDER = LABEL_FOLDER_COLORED
            file_tp, file_fp, file_conf, file_ground_truth_color_counter, file_prediction_color_counter, file_sheep_count = get_metrics(fileroot, partition_coordinates, use_ir, show_image, show_print, best_conf)
            for k,v in file_ground_truth_color_counter.items(): ground_truth_color_counter[k] += v
            for k,v in file_prediction_color_counter.items(): prediction_color_counter[k] += v

            SELECTED_LABEL_FOLDER = LABEL_FOLDER_OBSCURED
            file_tp, file_fp, file_conf, file_ground_truth_obscured_counter, file_prediction_obscured_counter, file_sheep_count = get_metrics(fileroot, partition_coordinates, use_ir, show_image, show_print, best_conf)
            for k,v in file_ground_truth_obscured_counter.items(): ground_truth_obscured_counter[k] += v
            for k,v in file_prediction_obscured_counter.items(): prediction_obscured_counter[k] += v

            file_grid_tp, file_grid_tn, file_grid_fp, file_grid_fn, file_grid_total_sheep_count, file_grid_found_sheep_count, file_grid_ground_truths, file_grid_confidences = get_grid_metrics(fileroot, partition_coordinates, use_ir, show_image, show_print, best_conf)
            grid_tp += file_grid_tp
            grid_tn += file_grid_tn
            grid_fp += file_grid_fp
            grid_fn += file_grid_fn
            grid_total_sheep_count += file_grid_total_sheep_count
            grid_found_sheep_count += file_grid_found_sheep_count
            grid_ground_truths = np.concatenate([grid_ground_truths, file_grid_ground_truths])
            grid_confidences = np.concatenate([grid_confidences, file_grid_confidences])


        except FileNotFoundError as e:
            print(e)
            # I had some missing labels... :/
    
    grid_precision = grid_tp / (grid_tp + grid_fp) if grid_tp > 0 else 0
    grid_recall = grid_tp / (grid_tp + grid_fn) if grid_tp > 0 else 0

    grid_sheep_recall = grid_found_sheep_count / grid_total_sheep_count

    # This is the one Kari used in her metrics
    grid_sklearn_aps = average_precision_score(grid_ground_truths, grid_confidences)

    print("\nGRID METRICS")
    print("sklearn_aps:", grid_sklearn_aps)
    print("precision:", grid_precision)
    print("recall:", grid_recall)
    print("sheep_recall:", grid_sheep_recall)

    results = [ap50s, best_conf, precision, recall, grid_sklearn_aps, grid_precision, grid_recall, grid_sheep_recall]
    results = ', '.join(map(lambda x: str(x), results))
    print(results)
    
    print("\nCATEGORY METRICS")
    cats = [best_conf]
    SELECTED_LABEL_CATEGORIES = LABEL_CATEGORIES_OBSCURED
    for cat in [0,3,2,1]:
        count = ground_truth_obscured_counter[cat]
        prediction_count = prediction_obscured_counter[cat]
        prediction_percentage = round(100 * prediction_count / count, 1)
        print(f"Category '{SELECTED_LABEL_CATEGORIES[cat]}' - Recall: {prediction_percentage}% ({prediction_count} / {count})")
        cats.append(prediction_count / count)

    SELECTED_LABEL_CATEGORIES = LABEL_CATEGORIES_COLORED
    for cat in [3,2,0,1]:
        count = ground_truth_color_counter[cat]
        prediction_count = prediction_color_counter[cat]
        prediction_percentage = round(100 * prediction_count / count, 1)
        print(f"Category '{SELECTED_LABEL_CATEGORIES[cat]}' - Recall: {prediction_percentage}% ({prediction_count} / {count})")
        cats.append(prediction_count / count)

    cats = ', '.join(map(lambda x: str(x), cats))
    print(cats)




def calculate_metrics_for_confidences():
    # get_metrics("2019_08_storli1_0720", partition_coordinates=None, use_ir=False, show_image=True)
    confidences = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.8, 0.9]
    
    conf_ap = []
    conf_ap50s = []
    conf_precision = []
    conf_recall = []
    conf_sheep_recall = []

    for conf in confidences:
        # LabelSet.label_confidence_threshold = conf
        # print("\nThreshold for label confidence:", LabelSet.label_confidence_threshold)

        sklearn_aps, ap50s, precision, recall, total_sheep_count_sum, found_sheep_count_sum = calculate_metrics()
        conf_ap.append(sklearn_aps)
        conf_ap50s.append(ap50s)
        conf_precision.append(precision)
        conf_recall.append(recall)
        conf_sheep_recall.append(found_sheep_count_sum / total_sheep_count_sum)

    plt.plot(confidences, conf_ap, label="ap")
    plt.plot(confidences, conf_ap50s, label="ap50s")
    plt.plot(confidences, conf_precision, label="precision")
    plt.plot(confidences, conf_recall, label="recall")
    plt.plot(confidences, conf_sheep_recall, label="sheep_recall")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # calculate_metrics_for_confidences()
    calculate_metrics(partition_coordinates=None, use_ir=True, show_image=False, show_print=False)
    # get_metrics('2021_09_holtan_1210', use_ir=True, show_image=True, show_print=True)

