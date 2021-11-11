import { readdir, rename} from 'fs/promises';
import { readFileSync, writeFileSync, readFile, writeFile } from 'fs';
import sizeOf from 'image-size'
import path from 'path';

// HELPERS

const LABELS_PATH = path.resolve('./roboflow_labels/')

// SCRIPT

const labels = await readdir(LABELS_PATH)

const categoriesConverter = { 
  '0': '0', 
  '1': '0', 
  '2': '0', 
  '3': '0', 
}

let distribution = {
  '0': 0,
  '1': 0,
  '2': 0,
}

function renameCategoryId(name) {
  try {
    const data = readFileSync(LABELS_PATH + '/' + name, 'utf8')
    // Split into rows
    let rows = data.split('\n')
  
    // Also split into columns
    rows = rows.map(r => r.split(' '))
    
    // Find old ids and new converted ids
    const oldIds = rows.map(r => r[0])
    const newIds = oldIds.map(id => categoriesConverter[id])

    newIds.forEach(id => distribution[id]++)
  
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



for (const fileName of labels) {
  renameCategoryId(fileName)
}
  
console.log('New distribution: ', distribution)


