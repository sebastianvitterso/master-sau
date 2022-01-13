import { readFile, writeFile, readdir, rename } from 'fs/promises';
import path from 'path';

// HELPERS


const BASE_PATH = path.resolve('../../data-cropped-partitioned/train/')

const FROM_PATH = path.resolve(BASE_PATH + '/images/')
const LABELS_PATH = path.resolve(BASE_PATH + '/labels/')
const TO_PATH = path.resolve(BASE_PATH + '/images_unlabeled/')

// SCRIPT

const images = await readdir(FROM_PATH)
const labels = await readdir(LABELS_PATH)

const labelsLookUptable = labels.reduce(function(map, fileName) {
  const label = fileName.split('.')[0]
  map[label] = true;
  return map;
}, {});


let movedCount = 0
let processedCount = 0
let totalCount = images.length

for (const fileName of images) {
  processedCount++
  const name = fileName.split('.')[0]

  if(!labelsLookUptable[name]) {
    movedCount++
    console.log(`Removed: ${movedCount} | ${processedCount} / ${totalCount}`)
    // console.log(`${FROM_PATH}/${fileName}`, `${TO_PATH}/${fileName}`)
    await rename(`${FROM_PATH}/${fileName}`, `${TO_PATH}/${fileName}`)

  }

}