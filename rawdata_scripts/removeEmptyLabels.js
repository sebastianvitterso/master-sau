import { readdir, rename} from 'fs/promises';
import { readFileSync, writeFileSync, readFile, writeFile, unlinkSync} from 'fs';
import path from 'path';

// HELPERS

const LABELS_PATH = path.resolve('../raw_data/hallvard/labels_with_color/')

// SCRIPT

const labels = await readdir(LABELS_PATH)

function removeLabelIfEmpty(name) {
  try {
    const data = readFileSync(LABELS_PATH + '/' + name, 'utf8')
    if (data == '') {
      console.log(name)
      unlinkSync(LABELS_PATH + '/' + name)
    }
  } catch (err) {
    console.error(err)
  }
}


for (const fileName of labels) {
  removeLabelIfEmpty(fileName)
}