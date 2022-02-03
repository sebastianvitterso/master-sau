import { readFile, writeFile, readdir, rename } from 'fs/promises';
import path from 'path';

// HELPERS

const LABELS_PATH = path.resolve('../../data/validation/color_labels/')

function fullPath(filename, basePath=LABELS_PATH) {
  return `${basePath}\\${filename}`
}

function removeRoboflowHash(fileName) {
  const parts = fileName.split('.')
  const name = parts[0].replace('_jpg', '').replace('_JPG', '') + '.' + parts[parts.length - 1]
  return name
}


// SCRIPT

const labels = await readdir(LABELS_PATH)

for (const fileName of labels) {
  const name = removeRoboflowHash(fileName)

  const oldPath = fullPath(fileName, LABELS_PATH)
  const newPath = fullPath(name, LABELS_PATH)
  
  console.log(oldPath, newPath)
  await rename(oldPath, newPath)

}
