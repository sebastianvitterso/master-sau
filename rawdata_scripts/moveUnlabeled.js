import { readFile, writeFile, readdir, rename } from 'fs/promises';
import path from 'path';

// HELPERS

const FROM_PATH = path.resolve('../raw_data/hallvard/IR/')
const LABELS_PATH = path.resolve('../raw_data/hallvard/labels/')
const TO_PATH = path.resolve('../raw_data/hallvard/IR_unlabeled/')

// SCRIPT

const images = await readdir(FROM_PATH)
const labels = await readdir(LABELS_PATH)

const labelsLookUptable = labels.reduce(function(map, fileName) {
  const label = fileName.split('.')[0]
  map[label] = true;
  return map;
}, {});


let count = 0
for (const fileName of images) {
  const name = fileName.split('.')[0]

  if(!labelsLookUptable[name]) {
    count++
    console.log(`${FROM_PATH}/${fileName}`, `${TO_PATH}/${fileName}`)
    await rename(`${FROM_PATH}/${fileName}`, `${TO_PATH}/${fileName}`)

  }

}

console.log(`Unlabeled files count: ${count}/${images.length}`)
