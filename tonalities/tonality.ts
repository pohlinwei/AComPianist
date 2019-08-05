export class Tonality {
    private dominantHue: number;

    constructor(dominantHue: number) {
        this.dominantHue = dominantHue;
    }

    // to be overridden
    changeKey(hue: number) {
        return [hue];
    }

    getDominantHue() {
        return this.dominantHue;
    }
}