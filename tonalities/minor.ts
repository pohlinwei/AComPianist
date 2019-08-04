/**
 * The following corresponds to the frequency of
 * ['c', 'cs', 'd', 'ds', 'e', 'f', 'fs', 'g', 'gs', 'a', 'as', 'b'] 
 */
 const allKeyIndices = [
    // d minor
    [0, 1, 2, 0, 1, 1, 0, 1, 0, 1, 1, 0], 
    // a minor
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 2, 0, 1],
    // e minor
    [1, 0, 0, 1, 2, 0, 1, 1, 0, 1, 0, 1],
    // b minor
    [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 2],
    // f# minor
    [0, 1, 1, 0, 0, 1, 2, 0, 1, 1, 0, 1],
    // c# minor
    [1, 2, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    // g# minor
    [0, 1, 0, 1, 1, 0, 0, 1, 2, 0, 1, 1],
    // d# minor
    [0, 0, 1, 2, 0, 1, 1, 0, 1, 0, 1, 1],
    // a# minor
    [1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 2, 0],
    // f minor
    [1, 1, 0, 0, 1, 2, 0, 1, 1, 0, 1, 0],
    // c minor
    [2, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    // g minor
    [1, 0, 1, 1, 0, 0, 1, 2, 0, 1, 1, 0]
]
const NUM_KEYS = 12;
const SECTOR_DEGREE = 360 / NUM_KEYS;

export class Minor {
    private dominantHue: number;

    constructor(dominantHue: number) {
        this.dominantHue = dominantHue;
    }

    changeKey(hue: number) {
        // determine absolute change in hue
        let hueChange = hue - this.dominantHue;
        if (hueChange < 0) {
            hueChange += 360;
        }
        let hueChangeIndex = Math.floor(hueChange / SECTOR_DEGREE);
        return allKeyIndices[hueChangeIndex];
    }

    getDominantHue() {
        return this.dominantHue;
    }
}