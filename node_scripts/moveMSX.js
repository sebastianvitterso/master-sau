import { readFile, writeFile, readdir, rename } from 'fs/promises';
import { existsSync, mkdirSync } from 'fs';
import path from 'path';

// HELPERS


const BASE_PATH = path.resolve('../../data-cropped-partitioned-no-msx-test/train/')

const FROM_PATH = path.resolve(BASE_PATH + '/images/')
const MSX_FILENAMES = String(await readFile(path.resolve('msx_ir.txt'))).split('\n').map(line => line.trim())
const TO_PATH = path.resolve(BASE_PATH + '/images_msx/')

if (!existsSync(TO_PATH)){
  mkdirSync(TO_PATH);
}

// SCRIPT

const images = await readdir(FROM_PATH)

let movedCount = 0
let processedCount = 0
let totalCount = images.length

images.sort()
for (const fileName of images) {
  processedCount++
  const imageName = fileName.split('_p')[0] + '.JPG'

  if(MSX_FILENAMES.includes(imageName)) {
    movedCount++
    console.log(`Moved: ${movedCount} | ${processedCount} / ${totalCount}`)
    console.log(`${FROM_PATH}/${fileName}`, `${TO_PATH}/${fileName}`)
    await rename(`${FROM_PATH}/${fileName}`, `${TO_PATH}/${fileName}`)

  }

}