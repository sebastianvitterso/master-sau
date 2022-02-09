import { readdir} from 'fs/promises';
import { readFileSync, unlinkSync, existsSync, renameSync} from 'fs';

const DELETE_LABEL = true

const BASE_FOLDER = '../../'

const DATA_FOLDER = BASE_FOLDER + 'data/' // Should probably not be run on this folder.
const DATA_CROPPED_FOLDER = BASE_FOLDER + 'data-cropped/' // Should probably not be run on this folder.
const DATA_CROPPED_PARTITIONED_FOLDER = BASE_FOLDER + 'data-cropped-partitioned/'
const DATA_PARTITIONED_FOLDER = BASE_FOLDER + 'data-partitioned/'
// YOU CAN CHANGE THIS!
const SELECTED_DATA_FOLDER = DATA_FOLDER // e.g. '../../data-partitioned/'


const DATA_SET_TRAIN_FOLDER = SELECTED_DATA_FOLDER + 'train/'
const DATA_SET_VALIDATION_FOLDER = SELECTED_DATA_FOLDER + 'validation/'
// CHANGE THIS, IF YOU HAVEN'T JUST MOVED ALL YOUR IMAGES BACK INTO THE 'train'-FOLDER!
const SELECTED_DATA_SET_FOLDER = DATA_SET_TRAIN_FOLDER // e.g. '../../data/train/'


const RGB_FOLDER = SELECTED_DATA_SET_FOLDER + 'images/'
const IR_FOLDER = SELECTED_DATA_SET_FOLDER + 'ir/'
const UNLABELED_RGB_FOLDER = SELECTED_DATA_SET_FOLDER + 'images_unlabeled/'
const UNLABELED_IR_FOLDER = SELECTED_DATA_SET_FOLDER + 'ir_unlabeled/'
const LABEL_FOLDER = SELECTED_DATA_SET_FOLDER + 'labels/'


let removedLabels = 0
let removedRgb = 0
let removedIr = 0
const labels = await readdir(LABEL_FOLDER)

const nonEmptyLabels = new Set()

for (const [i, fileName] of labels.entries()) {
  try {
    const data = readFileSync(LABEL_FOLDER + fileName, 'utf8')
    
    if (data == '') { // the label is empty - time to delete the corresponding images!
      if(DELETE_LABEL) {
        unlinkSync(LABEL_FOLDER + fileName)
        removedLabels++
      }
    } 
    else {
      const fileRoot = fileName.split('.')[0]
      nonEmptyLabels.add(fileRoot)
    }
    console.log(`Removed labels: ${removedLabels} | ${i + 1} / ${labels.length}`)
  } catch (err) {
    console.error(err)
  }
}

const rgb_images = await readdir(RGB_FOLDER)
for (const [i, fileName] of rgb_images.entries()) {
  try {
    const fileRoot = fileName.split('.')[0]

    if (!nonEmptyLabels.has(fileRoot)) {
      // delete the corresponding RGB image
      const rgbOldName = RGB_FOLDER + fileRoot + '.JPG'
      const rgbNewName = UNLABELED_RGB_FOLDER + fileRoot + '.JPG'
      if(existsSync(rgbOldName)) {
        renameSync(rgbOldName, rgbNewName)
        removedRgb++
      }
    }
    console.log(`Moved RGB: ${removedRgb} | ${i + 1} / ${rgb_images.length}`)
  } catch (err) {
    console.error(err)
  }
}

const ir_images = await readdir(IR_FOLDER)
for (const [i, fileName] of ir_images.entries()) {
  try {
    const fileRoot = fileName.split('.')[0]

    if (!nonEmptyLabels.has(fileRoot)) {
      // delete the corresponding RGB image
      const irOldName = IR_FOLDER + fileRoot + '.JPG'
      const irNewName = UNLABELED_IR_FOLDER + fileRoot + '.JPG'
      if(existsSync(irOldName)) {
        renameSync(irOldName, irNewName)
        removedIr++
      }
    }
    console.log(`Moved IR: ${removedIr} | ${i + 1} / ${ir_images.length}`)
  } catch (err) {
    console.error(err)
  }
}