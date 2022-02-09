import { readFile, writeFile, readdir, rename } from 'fs/promises';
import sizeOf from 'image-size'
import path from 'path';

// HELPERS

const FROM_PATH = path.resolve('../data/train/images/')
const TO_PATH = path.resolve('../data/validation/images/')

// SCRIPT

const images = await readdir(FROM_PATH)
const MOVE_COUNT = 100

const shuffled = images.sort(() => 0.5 - Math.random());
let selectedImages = shuffled.slice(0, MOVE_COUNT);

for (const fileName of selectedImages) {

  console.log(`${FROM_PATH}/${fileName}`, `${TO_PATH}/${fileName}`)
  await rename(`${FROM_PATH}/${fileName}`, `${TO_PATH}/${fileName}`)

}
