import { readFile, writeFile, readdir, rename } from 'fs/promises';
import path from 'path';

// HELPERS

const FROM_PATH = path.resolve('./roboflow_images/')
const LABELS_PATH = path.resolve('./roboflow_labels/')
const TO_PATH = path.resolve('./output/labels/')

function fullPath(filename, basePath=FROM_PATH) {
  return `${basePath}\\${filename}`
}

function removeRoboflowHash(fileName) {
  const parts = fileName.split('.')
  const name = parts[0].replace('_JPG', '') + '.' + parts[parts.length - 1]
  return name
}


// SCRIPT

const images = await readdir(FROM_PATH)
const labels = await readdir(LABELS_PATH)

for (const fileName of images) {
  const name = removeRoboflowHash(fileName)

  const oldPath = fullPath(fileName, FROM_PATH)
  const newPath = fullPath(name, FROM_PATH)
  
  // console.log(oldPath, newPath)
  rename(oldPath, newPath)

}



