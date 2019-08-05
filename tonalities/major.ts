import { Tonality } from './tonality';

/**
 * The following corresponds to the frequency of
 * ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'b'] 
 */

const allKeyIndices = [
    // c major
    [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], 
    // g major
    [1, 0, 1, 0, 1, 0, 1, 2, 0, 1, 0, 1],
    // d major
    [0, 1, 2, 0, 1, 0, 1, 1, 0, 1, 0, 1,],
    // a major
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1],
    // e major
    [0, 1, 0, 1, 2, 0, 1, 0, 1, 1, 0, 1],
    // b major
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 2],
    // f# major
    [0, 1, 0, 1, 0, 1, 2, 0, 1, 0, 1, 1],
    // c# major
    [1, 2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
    // g# major
    [1, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1, 0],
    // d# major
    [1, 0, 1, 2, 0, 1, 0, 1, 1, 0, 1, 0],
    // a# major
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 2, 0],
    // f major
    [1, 0, 1, 0, 1, 2, 0, 1, 0, 1, 1, 0]
]
const NUM_KEYS = 12;
const SECTOR_DEGREE = 360 / NUM_KEYS;

export class Major extends Tonality {
    constructor(dominantHue: number) {
       super(dominantHue);
    }

    changeKey(hue: number) {
        // determine absolute change in hue
        let hueChange = hue - this.getDominantHue();
        if (hueChange < 0) {
            hueChange += 360;
        }
        let hueChangeIndex = Math.floor(hueChange / SECTOR_DEGREE);
        return allKeyIndices[hueChangeIndex];
    }
}