import { readdir, rename} from 'fs/promises';
import { readFileSync, writeFileSync, readFile, writeFile, unlinkSync} from 'fs';
import path from 'path';

// HELPERS

const BASE_PATH = path.resolve('../../master-sau/data-partitioned/train/')


const LABELS_PATH = path.resolve(BASE_PATH + '/labels/')

// SCRIPT

const labels = await readdir(LABELS_PATH)

let removedCount = 0
let processedCount = 0
let totalCount = labels.length

function removeLabelIfEmpty(name) {
  try {
    const data = readFileSync(LABELS_PATH + '/' + name, 'utf8')
    if (data == '') {
      removedCount++
      console.log(`Removed: ${removedCount} | ${processedCount} / ${totalCount}`)
      unlinkSync(LABELS_PATH + '/' + name)
    }
  } catch (err) {
    console.error(err)
  }
}


for (const fileName of labels) {
  processedCount++
  removeLabelIfEmpty(fileName)
}