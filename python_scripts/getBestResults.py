import csv

folder = 'rgb'
filename = f'../yolov5/runs/train/{folder}/results.csv'

with open(filename, newline='') as csvfile:
    max_map_05 = (0,0)
    max_map_05_095 = (0,0)
    max_precision = (0,0)
    max_recall = (0,0)

    headers = [header.strip() for header in csvfile.readline().split(',')]
    reader = csv.DictReader(csvfile, fieldnames=headers)
    rows = list(reader)
    epoch_count = len(rows)

    for row in rows:
        if(float(row['metrics/mAP_0.5']) > max_map_05[0]):
            max_map_05 = ( float(row['metrics/mAP_0.5']), int(row['epoch']) )

        if(float(row['metrics/mAP_0.5:0.95']) > max_map_05_095[0]):
            max_map_05_095 = ( float(row['metrics/mAP_0.5:0.95']), int(row['epoch']) )

        if(float(row['metrics/precision']) > max_precision[0]):
            max_precision = ( float(row['metrics/precision']), int(row['epoch']) )

        if(float(row['metrics/recall']) > max_recall[0]):
            max_recall = ( float(row['metrics/recall']), int(row['epoch']) )


    print(f'==========================')
    print(f'{filename}')
    print(f'epoch count: {epoch_count}')
    print(f'max mAP_0.5: {max_map_05[0]} ({max_map_05[1]})')
    print(f'max mAP_0.5:0.95: {max_map_05_095[0]} ({max_map_05_095[1]})')
    print(f'max precision: {max_precision[0]} ({max_precision[1]})')
    print(f'max recall: {max_recall[0]} ({max_recall[1]})')

    print(folder, epoch_count, max_map_05[0], max_map_05[1], max_map_05_095[0], max_map_05_095[1], max_precision[0], max_precision[1], max_recall[0], max_recall[1])
