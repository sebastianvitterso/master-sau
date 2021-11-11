import { readFile, writeFile, readdir, rename } from 'fs/promises';
import sizeOf from 'image-size'
import path from 'path';

// HELPERS

const FROM_PATH = path.resolve('./output/labels')
function fullPath(filename, basePath=FROM_PATH) {
  return `${basePath}\\${filename}`
}

function root(filename) {
  return filename.split('.')[0]
}

// SCRIPT

const transforms = JSON.parse(await readFile(fullPath('transforms_combined.json', './'), 'utf-8'))
const files = await readdir(FROM_PATH)

for (const filename of files) {
  if(!filename.includes('.txt')) {
    continue
  }

  const fileRoot = root(filename)
  const newName = transforms[fileRoot] + '.txt'

  const oldPath = fullPath(filename)
  const newPath = fullPath(newName)
  
  // console.log(oldPath, newPath)
  // continue
  rename(oldPath, newPath)
}
