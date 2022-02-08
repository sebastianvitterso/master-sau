import { readFile, writeFile, readdir, rename } from 'fs/promises';
import sizeOf from 'image-size'
import path from 'path';

// HELPERS

const FROM_PATH = path.resolve('../raw_data/hallvard/102MEDIA/')
const TO_PATH_IR = path.resolve('./output/IR')
const TO_PATH_RGB = path.resolve('./output/RGB')

function fullPath(filename, basePath=FROM_PATH) {
  return `${basePath}\\${filename}`
}

const transforms = {}
function storeTransform(from, to) {
  transforms[root(from)] = root(to)
}

function root(filename) {
  return filename.split('.')[0]
}

function checkRGB(filename) {
  const dimensions = sizeOf(fullPath(filename))
  return dimensions.width > 2000
}

function formatAsIR(filename) { 
  //filename should be on form 2020_05_orkanger_0690.JPG
  const fileroot = root(filename)
  const oldNum =  +(fileroot.slice(-4))
  const newNum = oldNum - 1
  return filename.replace(oldNum, newNum)
}


// SCRIPT

const images = await readdir(FROM_PATH)
for (const image of images) {
  if(!image.includes('.JPG')) {
    continue
  }
  const oldName = image
  const isRGB = checkRGB(image)

  let newName = oldName                                             // may20_100MEDIA_DJI_0690.JPG
  // newName = newName.replace('aug19_10', '2019_08_storli1_')         // 2020_05_orkanger_0MEDIA_DJI_0690.JPG
  newName = newName.replace('DJI_0', '2021_10_holtan_2')                      // 2020_05_orkanger_0690.JPG

  if(!isRGB) {
    newName = formatAsIR(newName)                                   // 2020_05_orkanger_0689_IR.JPG
  }
  
  // storeTransform(oldName, newName)
  
  const oldPath = fullPath(oldName)
  const newPath = fullPath(newName, isRGB ? TO_PATH_RGB : TO_PATH_IR)
  
  // console.log(oldPath, newPath)
  // continue
  await rename(oldPath, newPath)
}

// await writeFile(`${FROM_PATH}\\transforms.json`, JSON.stringify(transforms))
