import { readdir, rename} from 'fs/promises';
import { readFileSync, writeFileSync, readFile, writeFile, unlinkSync} from 'fs';
import sizeOf from 'image-size'
import path from 'path';

// HELPERS

const FROM_PATH = path.resolve('./roboflow_images/')
const LABELS_PATH = path.resolve('./roboflow_labels/')
const TO_PATH = path.resolve('./output/labels/')

// SCRIPT

const images = await readdir(FROM_PATH)
const labels = await readdir(LABELS_PATH)

function removeLabelIfEmpty(name) {
  try {
    const data = readFileSync(LABELS_PATH + '/' + name, 'utf8')
    if (data == '') {
      unlinkSync(LABELS_PATH + '/' + name)
    }
  } catch (err) {
    console.error(err)
  }
}


for (const fileName of labels) {
  removeLabelIfEmpty(fileName)
}