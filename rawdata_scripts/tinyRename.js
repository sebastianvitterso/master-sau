import { readFile, writeFile, readdir, rename } from 'fs/promises';
import path from 'path';

// HELPERS

const BASE_PATH = path.resolve('./old_labels/')
function fullPath(filename) {
  return `${BASE_PATH}\\${filename}`
}

// SCRIPT

const files = (await readdir(BASE_PATH)) //.filter(file => file.includes('hallvard'))

for (const oldName of files) {
  const newName = oldName.replace('DJI_0', '2021_09_holtan_1')
  
  // console.log(oldName, newName)
  await rename(fullPath(oldName), fullPath(newName))
}

