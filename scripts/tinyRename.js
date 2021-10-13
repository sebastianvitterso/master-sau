import { readFile, writeFile, readdir, rename } from 'fs/promises';
import path from 'path';

// HELPERS

const BASE_PATH = path.resolve('../raw_data/Lønset/RGB_unlabeled/')
function fullPath(filename) {
  return `${BASE_PATH}\\${filename}`
}

// SCRIPT

const files = (await readdir(BASE_PATH)) //.filter(file => file.includes('hallvard'))

for (const oldName of files) {
  const newName = oldName.replace('DJI_', '2018_10_lonset_')
  
  console.log(oldName, newName)
  // continue
  await rename(fullPath(oldName), fullPath(newName))
}

