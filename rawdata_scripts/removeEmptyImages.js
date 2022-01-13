import { readdir} from 'fs/promises';
import { readFileSync, unlinkSync, existsSync} from 'fs';

const BASE_FOLDER = '../../'

const DATA_FOLDER = BASE_FOLDER + 'data/' // Should probably not be run on this folder.
const DATA_CROPPED_FOLDER = BASE_FOLDER + 'data-cropped/' // Should probably not be run on this folder.
const DATA_CROPPED_PARTITIONED_FOLDER = BASE_FOLDER + 'data-cropped-partitioned/'
const DATA_PARTITIONED_FOLDER = BASE_FOLDER + 'data-partitioned/'
// YOU CAN CHANGE THIS!
const SELECTED_DATA_FOLDER = DATA_PARTITIONED_FOLDER // e.g. '../../data-partitioned/'


const DATA_SET_TRAIN_FOLDER = SELECTED_DATA_FOLDER + 'train/'
const DATA_SET_VALIDATION_FOLDER = SELECTED_DATA_FOLDER + 'validation/'
// CHANGE THIS, IF YOU HAVEN'T JUST MOVED ALL YOUR IMAGES BACK INTO THE 'train'-FOLDER!
const SELECTED_DATA_SET_FOLDER = DATA_SET_TRAIN_FOLDER // e.g. '../../data/train/'


const RGB_FOLDER = SELECTED_DATA_SET_FOLDER + 'rgb/'
const IR_FOLDER = SELECTED_DATA_SET_FOLDER + 'ir/'
const LABEL_FOLDER = SELECTED_DATA_SET_FOLDER + 'labels/'


let removedLabels = 0
let removedRgb = 0
let removedIr = 0
const labels = await readdir(LABEL_FOLDER)

for (const [i, fileName] of labels.entries()) {
  try {
    const data = readFileSync(LABEL_FOLDER + fileName, 'utf8')
    if (data == '') { // the label is empty - time to delete the corresponding images!
      const fileRoot = fileName.split('.')[0]
      
      // delete the corresponding RGB image
      const rgbFileName = RGB_FOLDER + fileRoot + '.JPG'
      if(existsSync(rgbFileName)) {
        unlinkSync(rgbFileName)
        removedRgb++
      }
      
      // delete the corresponding IR image
      const irFileName = IR_FOLDER + fileRoot + '.JPG'
      if(existsSync(irFileName)) {
        unlinkSync(irFileName)
        removedIr++
      }
    }
    console.log(`Removed: Labels: ${removedLabels}, RGB: ${removedRgb}, IR: ${removedIr} | ${i + 1} / ${labels.length}`)
  } catch (err) {
    console.error(err)
  }
}
