"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var allKeyIndices = [
    [0, 1, 2, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 2, 0, 1],
    [1, 0, 0, 1, 2, 0, 1, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 2],
    [0, 1, 1, 0, 0, 1, 2, 0, 1, 1, 0, 1],
    [1, 2, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 1, 0, 0, 1, 2, 0, 1, 1],
    [0, 0, 1, 2, 0, 1, 1, 0, 1, 0, 1, 1],
    [1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 2, 0],
    [1, 1, 0, 0, 1, 2, 0, 1, 1, 0, 1, 0],
    [2, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 2, 0, 1, 1, 0]
];
var NUM_KEYS = 12;
var SECTOR_DEGREE = 360 / NUM_KEYS;
var Minor = (function () {
    function Minor(dominantHue) {
        this.dominantHue = dominantHue;
    }
    Minor.prototype.changeKey = function (hue) {
        var hueChange = hue - this.dominantHue;
        if (hueChange < 0) {
            hueChange += 360;
        }
        var hueChangeIndex = Math.floor(hueChange / SECTOR_DEGREE);
        return allKeyIndices[hueChangeIndex];
    };
    Minor.prototype.getDominantHue = function () {
        return this.dominantHue;
    };
    return Minor;
}());
exports.Minor = Minor;
