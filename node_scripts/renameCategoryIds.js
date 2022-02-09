import { readdir, rename} from 'fs/promises';
import { readFileSync, writeFileSync, readFile, writeFile } from 'fs';
import path from 'path';

// HELPERS

const LABELS_PATH = path.resolve('../../data/validation/occlusion_labels/labels/')

// SCRIPT

const categoriesConverter = { 
  '0': '0',
  '1': '0',
  '2': '0',
  '3': '0',
  '4': '0',
  '5': '0',
  '6': '0',
}

let oldDistribution = {
  '0': 0,
  '1': 0,
  '2': 0,
  '3': 0,
  '4': 0,
  '5': 0,
  '6': 0,
}
let newDistribution = {
  '0': 0,
  '1': 0,
  '2': 0,
  '3': 0,
  '4': 0,
  '5': 0,
  '6': 0,
}

function renameCategoryId(name) {
  try {
    const data = readFileSync(LABELS_PATH + '/' + name, 'utf8')
    
    if (data == '') {
      return
    }
    
    // Split into rows
    let rows = data.split('\n')
  
    // Also split into columns
    rows = rows.map(r => r.split(' '))
    
    // Find old ids and new converted ids
    const oldIds = rows.map(r => r[0])
    const newIds = oldIds.map(id => categoriesConverter[id])

    oldIds.forEach(id => oldDistribution[id]++)
    newIds.forEach(id => newDistribution[id]++)
  
    // Replace old with new
    rows.forEach(r => r[0] = newIds.shift())
  
    // Make string output
    rows = rows.map(r => r.join(' '))
    const output = rows.join('\n')

    writeFileSync(LABELS_PATH + '/' + name, output)
  
  } catch (err) {
    console.error(err)
  }
}

// SCRIPT

const labels = await readdir(LABELS_PATH)

for (const fileName of labels) {
  renameCategoryId(fileName)
}
  
console.log('Old distribution: ', oldDistribution)
console.log('New distribution: ', newDistribution)


