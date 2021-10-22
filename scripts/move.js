import { readFile, writeFile, readdir, rename } from 'fs/promises';
import sizeOf from 'image-size'
import path from 'path';

// HELPERS

const FROM_PATH = path.resolve('./orkanger')
const TO_PATH_IR = path.resolve('./output/IR')
const TO_PATH_RGB = path.resolve('./output/RGB')
const IS_IR = filename => filename.includes('_IR')

// SCRIPT

const images = await readdir(FROM_PATH)
for (const oldName of images) {
  if(!oldName.includes('.JPG')) {
    continue
  }
  
  let newName = oldName
  newName = newName.replace('_IR', '')

  const isIR = IS_IR(oldName)
  const toPath = isIR ? TO_PATH_IR : TO_PATH_RGB

  rename(`${FROM_PATH}/${oldName}`, `${toPath}/${newName}`)
}
