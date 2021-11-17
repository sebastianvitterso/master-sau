import { readdir, rename } from 'fs/promises';
import { readFileSync, writeFileSync, readFile, writeFile } from 'fs';

import path from 'path';

// HELPERS

const FROM_PATH = path.resolve('./old_labels.json')
const TO_PATH = path.resolve('./new_labels.json')
const TRANSFORM_PATH = path.resolve('./transforms_combined.json')



// SCRIPT

const oldLabelData = readFileSync(FROM_PATH, 'utf8')
const transforms_data = readFileSync(TRANSFORM_PATH, 'utf8')

const transforms = JSON.parse(transforms_data)
let oldLabels = JSON.parse(oldLabelData)
let newLabels = {}


for (const oldFileName of Object.keys(oldLabels)) {
  const oldName = oldFileName.split('.')[0]
  const newName = transforms[oldName] + '.JPG'
  newLabels[newName] = oldLabels[oldFileName]
  
  console.log(oldName, newName)
}

const output = JSON.stringify(newLabels)
writeFileSync(TO_PATH, output)