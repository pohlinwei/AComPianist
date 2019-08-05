/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as tf from '@tensorflow/tfjs-core'
import { Minor } from './tonalities/minor';
import { Major } from './tonalities/major';
import { Tonality } from './tonalities/tonality';

///////////////////////////////////////////
/////     ALL HTML ELEMENTS      /////////
//////////////////////////////////////////
/* HTML elements for intro page */
const introPage = <HTMLDivElement> document.getElementById('intro-page');
const naviButton = <HTMLDivElement> document.getElementById('navi-button');

/* HTML elements for uploading of image */
const uploadPage = <HTMLDivElement> document.getElementById('upload-page')
const submitImage = <HTMLInputElement> document.getElementById('submit-image');
const file = <HTMLInputElement> document.querySelector('input[type=file]');
const form = <HTMLFormElement> document.querySelector('form');

/* HTML element for indicating that device is unsupported */
const notSupportedPage = <HTMLDivElement> document.getElementById('not-supported-page');

/* HTML element for loading */
const loadingPage = <HTMLDivElement> document.getElementById('loading-page');

/* HTML elements for main page */
const mainPage = <HTMLDivElement> document.getElementById('main');
const canvas = <HTMLCanvasElement> document.querySelector('canvas'); 
const densityDisplayVal = <HTMLParagraphElement> document.getElementById('note-density-val');
const gainDisplayVal = <HTMLParagraphElement> document.getElementById('gain-val');
const notes = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'];
const histogramDisplayElements: HTMLDivElement[] = notes.map(
    note => document.getElementById(note) as HTMLDivElement);
const guide = document.getElementById('guide');


///////////////////////////////////////////
/////     ALL GLOBAL VARIABLES    /////////
//////////////////////////////////////////
// rescale all uploaded images to this size
const IMG_SIZE = 256;

/* Variables for performance_rnn */
const Piano = require('tone-piano').Piano;

let lstmKernel1: tf.Tensor2D;
let lstmBias1: tf.Tensor1D;
let lstmKernel2: tf.Tensor2D;
let lstmBias2: tf.Tensor1D;
let lstmKernel3: tf.Tensor2D;
let lstmBias3: tf.Tensor1D;
let c: tf.Tensor2D[];
let h: tf.Tensor2D[];
let fcB: tf.Tensor1D;
let fcW: tf.Tensor2D;
const forgetBias = tf.scalar(1.0);
const activeNotes = new Map<number, number>();

// How many steps to generate per generateStep call.
// Generating more steps makes it less likely that we'll lag behind in note
// generation. Generating fewer steps makes it less likely that the browser UI
// thread will be starved for cycles.
const STEPS_PER_GENERATE_CALL = 10;
// How much time to try to generate ahead. More time means fewer buffer
// underruns, but also makes the lag from UI change to output larger.
const GENERATION_BUFFER_SECONDS = .5;
// If we're this far behind, reset currentTime time to piano.now().
const MAX_GENERATION_LAG_SECONDS = 1;
// If a note is held longer than this, release it.
const MAX_NOTE_DURATION_SECONDS = 3;

const NOTES_PER_OCTAVE = 12;
const PITCH_HISTOGRAM_SIZE = NOTES_PER_OCTAVE;

const RESET_RNN_FREQUENCY_MS = 30000;

// variables that (directly) influence music generation
let pitchHistogramEncoding: tf.Tensor1D;
let noteDensityEncoding: tf.Tensor1D;
let conditioned = true;

let currentPianoTimeSec = 0;

let currentVelocity = 100;

const MIN_MIDI_PITCH = 0;
const MAX_MIDI_PITCH = 127;
const VELOCITY_BINS = 32;
const MAX_SHIFT_STEPS = 100;
const STEPS_PER_SECOND = 100;

const MAX_GAIN = 200;
const MIN_GAIN = 10;
const DEAFULT_GAIN = 25;

// The unique id of the currently scheduled setTimeout loop.
let currentLoopId = 0;

// max height for each histogram bar
const MAX_HEIGHT = 30;

const EVENT_RANGES = [
  ['note_on', MIN_MIDI_PITCH, MAX_MIDI_PITCH],
  ['note_off', MIN_MIDI_PITCH, MAX_MIDI_PITCH],
  ['time_shift', 1, MAX_SHIFT_STEPS],
  ['velocity_change', 1, VELOCITY_BINS],
];

function calculateEventSize(): number {
    let eventOffset = 0;
    for (const eventRange of EVENT_RANGES) {
        const minValue = eventRange[1] as number;
        const maxValue = eventRange[2] as number;
        eventOffset += maxValue - minValue + 1;
    }
    return eventOffset;
}
  
const EVENT_SIZE = calculateEventSize();
const PRIMER_IDX = 355;  // shift 1s.
let lastSample = tf.scalar(PRIMER_IDX, 'int32');
  
// for loading piano and neural net for music generation
const piano = new Piano({velocities: 4}).toMaster();  
const SALAMANDER_URL = 'https://storage.googleapis.com/' +
    'download.magenta.tensorflow.org/demos/SalamanderPiano/';
const CHECKPOINT_URL = 'https://storage.googleapis.com/' +
    'download.magenta.tensorflow.org/models/performance_rnn/tfjs';

// model for music generation is initially not ready
let modelReady = false;

/* Parameter control */
// Note density
let controlDensity: number;
const DEFAULT_DENSITY = 2;
const DENSITY_BIN_RANGES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
const NUM_DENSITY = 4; // maximum density is 16
// Tonality
let pitches: number[]; 
let controlTonality: Tonality;
// Gain
let controlGain = DEAFULT_GAIN;

///////////////////////////////////////////
/////      COMPABILITY CHECK      ////////
//////////////////////////////////////////
const isDeviceSupported = tf.ENV.get('WEBGL_VERSION') >= 1;

///////////////////////////////////////////
/////   ADD HTML EVENT HANDLERS   ////////
//////////////////////////////////////////
/* initialising display */
// only show intro page initially
notSupportedPage.classList.add('hidden');
uploadPage.classList.add('hidden');
mainPage.classList.add('hidden');
loadingPage.classList.add('hidden');
mainPage.style.display = 'none';

/* For intro page */
naviButton.onclick = () => {
    if (!modelReady) {
        introPage.classList.add('hidden');
        if (isDeviceSupported) {
            // show upload page
            // hide not supported page
            uploadPage.classList.remove('hidden');
        } else {
            // show notsupported page
            notSupportedPage.classList.remove('hidden');
        }
        // change it to 'resume' button
        naviButton.innerHTML = 'Resume <i class="fas fa-angle-double-right"></i>';
    } else {
        // display main page
        mainPage.style.display = 'flex';
        // hide intro page
        introPage.classList.add('hidden');
    }
}

/* For upload page */
// do not allow submission when no image is uploaded
submitImage.disabled = true;
file.onchange = () => {
    // check whether file value is none
    let fileAbsent = (file.value == '');
    submitImage.disabled = fileAbsent;
}
form.onsubmit = (event) => {
    // prevent webpage from reloading upon form submission
    event.preventDefault();
    // hide upload page
    uploadPage.classList.add('hidden');
    // display loading page
    loadingPage.classList.remove('hidden');
    // send image for processing and load performance_rnn model
    Promise.all([loadImage(), loadModels()])
        .then(response => {
            // hide loading page
            loadingPage.classList.add('hidden');
            // show main page 
            mainPage.style.display = 'flex';
            // get parameters
            let parameters = <string> response[0];
            parameters = parameters.slice(1, parameters.length - 2);
            const positive = parameters.split(',')[0];
            const avg_hue = <number> parseFloat(parameters.split(',')[1]);
            // set param
            set_param(positive, avg_hue); 
            // generate music
            generateStep(currentLoopId);
        })
        .catch(err => {
            // !!! to-do: specify which process is causing an error
            console.log('loadImage or loadModels process failed.');
            console.log(err);
        });
}

/* For main page */
guide.onclick = () => {
    // hide main page 
    mainPage.style.display = 'none';
    // show intro page
    introPage.classList.remove('hidden');
}

/* For user control */
canvas.onmousemove = (event) => {
    // time when event occurs
    let eventTime = event.timeStamp;
    // new x-coordinate of cursor
    let cursorX = event.clientX;
    // new y-coordinate of cursor
    let cursorY = event.clientY;
    // canvas's top left coordinate
    let canvasTopLeftX = canvas.offsetLeft;
    let canvasTopLeftY = canvas.offsetTop;
    // cursor's position relative to canvas
    let cursorRelativeX = cursorX - canvasTopLeftX;
    let cursorRelativeY = cursorY - canvasTopLeftY;
    // determine changes in control gain
    console.log('change gain');
    controlGainFunc(eventTime, cursorX, cursorY);
    // calculate hsv
    console.log('calculate hsv');
    let hsl = calculateHsv(cursorRelativeX, cursorRelativeY);
    let h = hsl[0];
    let l = hsl[2];
    // determine changes in tonality
    console.log('tonality change');
    controlTonalityFunc(h);
    // determine changes in note density
    console.log('density change');
    controlDensityFunc(l); 
    console.log('update');
    updateConditioningParams();
}

canvas.onmouseleave = () => {
    // increase gain value if it is less than default gain
    controlGain = DEAFULT_GAIN;
    gainDisplayVal.innerHTML = controlGain.toString() + ' %';
    // change key to original key
    pitches = controlTonality.changeKey(controlTonality.getDominantHue());
}

///////////////////////////////////////////
/////  FUNC FOR INITIALISING MODELS   ////
//////////////////////////////////////////

/**
 * Loads image, sends image for processing by python script.
 * Returns a promise. Upon resolving, parameters used for music generation 
 * are returned.
 */
function loadImage() {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
    
        reader.addEventListener("load", () => {
            const imageDataUrl= <string> reader.result;
            // resize image, send it to python script for processing and 
            // determine parameters for performance-rnn
            resizeImage(imageDataUrl)
                .then((resizedImage) => {
                    const xhttp = new XMLHttpRequest();
                    xhttp.onreadystatechange = () => {
                        if (xhttp.readyState == 4 && xhttp.status == 200) {
                            const response: string = xhttp.response;
                            resolve(response);
                        } else if (xhttp.status >= 400) {
                            reject('Unable to process image.');
                        }
                    }
                    xhttp.open("POST", "http://localhost:3000/get_params", true);
                    xhttp.setRequestHeader("Content-Type", "application/json;charser=UTF-8");
                    const data = {image : resizedImage}; 
                    xhttp.send(JSON.stringify(data)); 
                })
                .catch(err => {
                    reject('loadImage was unsuccessful. ' + err);
                });
        });

        if (file.files != null) {
            reader.readAsDataURL(file.files[0]);
        } else {
            reject('No image input file.');
        }
    });
}

/** 
 * Resizes image before processing 
 * Returns a promise. Upon resolving, resized image's dataurl is returned.
 */ 
function resizeImage(imgDataUrl: string) {
    return new Promise((resolve, reject) => {
        const image = new Image();

        image.onload = () => {
            const ctx = <CanvasRenderingContext2D> canvas.getContext('2d');
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = "high";
            // resize image and make resized image available for display later
            ctx.drawImage(image, 0, 0, IMG_SIZE, IMG_SIZE);
            // convert image to dataurl format
            const resizedImage = canvas.toDataURL('image/jpeg');
            // return image's dataurl
            resolve(resizedImage);
        }
        image.onerror = () => {
            reject('Resize image error.');
        }

        // load original image
        image.src = imgDataUrl;
    });
}

/**
 * Loads models (piano and weights for neural net) for music generation.
 * Returns a promise which returns 'loaded' if the process is successful.
 */
function loadModels() {
    return new Promise((resolve, reject) => {
        piano.load(SALAMANDER_URL)
        .then(() => {
          return fetch(`${CHECKPOINT_URL}/weights_manifest.json`)
                       .then((response) => response.json())
                       .then(
                           (manifest: tf.WeightsManifestConfig) =>
                               tf.loadWeights(manifest, CHECKPOINT_URL));
        })
        .then((vars: {[varName: string]: tf.Tensor}) => {
            lstmKernel1 =
                vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'] as
                tf.Tensor2D;
            lstmBias1 = vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'] as
                tf.Tensor1D;
  
            lstmKernel2 =
                vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'] as
                tf.Tensor2D;
            lstmBias2 = vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'] as
                tf.Tensor1D;
  
            lstmKernel3 =
                vars['rnn/multi_rnn_cell/cell_2/basic_lstm_cell/kernel'] as
                tf.Tensor2D;
            lstmBias3 = vars['rnn/multi_rnn_cell/cell_2/basic_lstm_cell/bias'] as
                tf.Tensor1D;
  
            fcB = vars['fully_connected/biases'] as tf.Tensor1D;
            fcW = vars['fully_connected/weights'] as tf.Tensor2D;
            // indicate that model is ready
            modelReady = true;
            resetRnn();
            resolve(true);
        })
        .catch((err: Error) => {
            reject('loadModel was unsucessful. ' + err);
        })
    });
}

function resetRnn() {
    c = [
        tf.zeros([1, lstmBias1.shape[0] / 4]),
        tf.zeros([1, lstmBias2.shape[0] / 4]),
        tf.zeros([1, lstmBias3.shape[0] / 4]),
    ];
    h = [
        tf.zeros([1, lstmBias1.shape[0] / 4]),
        tf.zeros([1, lstmBias2.shape[0] / 4]),
        tf.zeros([1, lstmBias3.shape[0] / 4]),
    ];
    if (lastSample != null) {
        lastSample.dispose();
    }
    lastSample = tf.scalar(PRIMER_IDX, 'int32');
    currentPianoTimeSec = piano.now();
    currentLoopId++;
}

/**
 * Sets initial parameters for music generation.
 */
function set_param(positive: string, avg_hue: number) {
    if (positive == 'True') {
        controlTonality = new Major(avg_hue);
    } else {
        controlTonality = new Minor(avg_hue);
    }
    pitches = controlTonality.changeKey(controlTonality.getDominantHue());
    // set note density to default
    controlDensity = DEFAULT_DENSITY;
    // update display
    gainDisplayVal.innerHTML = controlGain.toString() + ' %';
    densityDisplayVal.innerHTML = controlDensity.toString();
    updateConditioningParams();
}

///////////////////////////////////////////
/////   FUNC FOR UPDATING DISPLAY    /////
//////////////////////////////////////////
function updateDisplayHistogram(hist: number[]) {
    for (let i = 0; i < hist.length; i++) {
        const freq: number = hist[i];
        const height = freq == 0 
            ? 0 
            : freq == 1 
                ? 0.5 * MAX_HEIGHT 
                : MAX_HEIGHT;
        histogramDisplayElements[i].style.height =
            (height).toString() + 'px';
    }
}

///////////////////////////////////////////
/////    FUNC FOR UPDATING PARAMS    /////
//////////////////////////////////////////
function updateConditioningParams() {
    let pitchHistogram = pitches;
    updateDisplayHistogram(pitchHistogram);
  
    if (noteDensityEncoding != null) {
        noteDensityEncoding.dispose();
        noteDensityEncoding = null;
    }
  
    const noteDensity = DENSITY_BIN_RANGES[controlDensity];
    // update display for note density
    densityDisplayVal.innerHTML = noteDensity.toString();
    // update display for gain
    gainDisplayVal.innerHTML = controlGain.toString() + ' %';

    noteDensityEncoding =
        tf.oneHot(
            tf.tensor1d([controlDensity + 1], 'int32'),
            DENSITY_BIN_RANGES.length + 1).as1D();
  
    if (pitchHistogramEncoding != null) {
        pitchHistogramEncoding.dispose();
        pitchHistogramEncoding = null;
    }
    const buffer = tf.buffer<tf.Rank.R1>([PITCH_HISTOGRAM_SIZE], 'float32');
    const pitchHistogramTotal = pitchHistogram.reduce((prev, val) => {
      return prev + val;
    });
    console.log(pitchHistogramTotal);
    for (let i = 0; i < PITCH_HISTOGRAM_SIZE; i++) {
        console.log('pitch i: ' + pitchHistogram[i]);
        buffer.set(pitchHistogram[i] / pitchHistogramTotal, i); 
    }
    pitchHistogramEncoding = buffer.toTensor();
}
  

/* allows user to control gain */  
// initial x-coordinate of cursor
let x0 = 0; 
// initial y-coordinate of cursor
let y0 = 0; 
// time when cursor was last hovered over the canvas
let cursorTime: number;

/* 'rate of change in cursor movement' determines gain */
function controlGainFunc(eventTime: number, cursorX: number, cursorY: number) {
    // new x-coordinate of cursor
    let x1 = cursorX;
    // new y-coordinate of cursor
    let y1 = cursorY;
    let dx = Math.abs(x1 - x0);
    let dy = Math.abs(y1 - y0);
    // update x-coordinate
    x0 = x1;
    // update y-coordinate
    y0 = y1;
    // time lapse between cursor events
    let dt = isNaN(eventTime - cursorTime) ? 1 : (eventTime - cursorTime);
    // difference in distance
    let ds = dx * dx + dy * dy;
    console.log("ds: " + ds);
    console.log("dt: " + dt);
    console.log(ds / dt);
    // rate of change
    let rate = ds / dt;
    // scale rate before using it to determine gain
    let scaledRate = rate < 10 ? rate  * 100 : rate * 11;
    // determine gain
    if (scaledRate < 5) {
        controlGain = MIN_GAIN;
    } else if (scaledRate > 200) {
        controlGain = MAX_GAIN;
    } else {
        controlGain = Math.floor(scaledRate);
    }
    // update cursorTime 
    cursorTime = eventTime;
}

/* hue of pixel cursor is currently pointing to affects tonality */
function controlTonalityFunc(hue: number) {
    pitches = controlTonality.changeKey(hue);
}

/* lightness of pixel cursor is currently pointin to affects note density */
function controlDensityFunc(lightness: number) {
    controlDensity = NUM_DENSITY - 
        Math.floor(lightness / (100 / NUM_DENSITY));
} 

function calculateHsv(cursorRelativeX: number, cursorRelativeY: number) {
    console.log(cursorRelativeX);
    console.log(cursorRelativeY);
    const ctx = canvas.getContext('2d');
    const pixel = ctx.getImageData(cursorRelativeX, cursorRelativeY, 1, 1);
    const data = pixel.data;
    const r = data[0];
    const g = data[1];
    const b = data[2];
    console.log('rgba value: ' + r + ' ' + g + ' ' + b);
    return rgbToHsv(r, g, b);
} 
  
function rgbToHsv(r: number, g: number, b: number) {
    // Reduce r, g, b to values between 0 and 1
    r /= 255;
    g /= 255;
    b /= 255;
  
    const cmin = Math.min(r, g, b);
    const cmax = Math.max(r, g, b);
    const d = cmax - cmin;
    let h: number;
    let s: number;
    let l: number;
  
    // determine value of hue
    switch(cmax) {
        case r: {
            if (d == 0) {
                h = 0;
                break;
            }
            h = ((g - b) / d) % 6;
            break;
        }
        case g: {
            h = (b - r) / d + 2;
            break;
        }
        case b: {
            h = (r - g) / d + 4;
        }
    }
  
    h = Math.round(h * 60);
    // ensures h is positive
    h += h < 0 ? 360 : 0;
  
    // determine lightness
    l = (cmax + cmin) / 2;
    // determine saturation
    s = d == 0 ? 0 : d / (1 - Math.abs(2 * l - 1));
  
    // convert s and l values to percentages
    s = +(s * 100).toFixed(1);
    l = +(l * 100).toFixed(1);
    console.log('hsl value: ' + h + ' ' + s + ' ' + l);
    return [h, s, l];
}


///////////////////////////////////////////
/////    FUNC FOR MUSIC GENERATION   /////
//////////////////////////////////////////  
function getConditioning(): tf.Tensor1D {
    return tf.tidy(() => {
      if (!conditioned) {
        // TODO(nsthorat): figure out why we have to cast these shapes to numbers.
        // The linter is complaining, though VSCode can infer the types.
        const size = 1 + (noteDensityEncoding.shape[0] as number) +
            (pitchHistogramEncoding.shape[0] as number);
        const conditioning: tf.Tensor1D =
            tf.oneHot(tf.tensor1d([0], 'int32'), size).as1D();
        return conditioning;
      } else {
        const axis = 0;
        const conditioningValues =
            noteDensityEncoding.concat(pitchHistogramEncoding, axis);
        return tf.tensor1d([0], 'int32').concat(conditioningValues, axis);
      }
    });
}
  
async function generateStep(loopId: number) {
    if (loopId < currentLoopId) {
      // Was part of an outdated generateStep() scheduled via setTimeout.
      return;
    }
  
    const lstm1 = (data: tf.Tensor2D, c: tf.Tensor2D, h: tf.Tensor2D) =>
        tf.basicLSTMCell(forgetBias, lstmKernel1, lstmBias1, data, c, h);
    const lstm2 = (data: tf.Tensor2D, c: tf.Tensor2D, h: tf.Tensor2D) =>
        tf.basicLSTMCell(forgetBias, lstmKernel2, lstmBias2, data, c, h);
    const lstm3 = (data: tf.Tensor2D, c: tf.Tensor2D, h: tf.Tensor2D) =>
        tf.basicLSTMCell(forgetBias, lstmKernel3, lstmBias3, data, c, h);
  
    let outputs: tf.Scalar[] = [];
    [c, h, outputs] = tf.tidy(() => {
        // Generate some notes.
        const innerOuts: tf.Scalar[] = [];
        for (let i = 0; i < STEPS_PER_GENERATE_CALL; i++) {
            // Use last sampled output as the next input.
            const eventInput = tf.oneHot(
                lastSample.as1D(), EVENT_SIZE).as1D();
            // Dispose the last sample from the previous generate call, since we
            // kept it.
            if (i === 0) {
            lastSample.dispose();
            }
            const conditioning = getConditioning();
            const axis = 0;
            const input = conditioning.concat(eventInput, axis).toFloat();
            const output =
                tf.multiRNNCell([lstm1, lstm2, lstm3], input.as2D(1, -1), c, h);
            c.forEach(c => c.dispose());
            h.forEach(h => h.dispose());
            c = output[0];
            h = output[1];
  
            const outputH = h[2];
            const logits = outputH.matMul(fcW).add(fcB);
  
            const sampledOutput = tf.multinomial(logits.as1D(), 1).asScalar();
  
            innerOuts.push(sampledOutput);
            lastSample = sampledOutput;
        }
        return [c, h, innerOuts] as [tf.Tensor2D[], tf.Tensor2D[], tf.Scalar[]];
    });
  
    for (let i = 0; i < outputs.length; i++) {
        playOutput(outputs[i].dataSync()[0]);
    }
  
    if (piano.now() - currentPianoTimeSec > MAX_GENERATION_LAG_SECONDS) {
        console.warn(
          `Generation is ${piano.now() - currentPianoTimeSec} seconds behind, ` +
          `which is over ${MAX_NOTE_DURATION_SECONDS}. Resetting time!`);
        currentPianoTimeSec = piano.now();
    }
    const delta = Math.max(
        0, currentPianoTimeSec - piano.now() - GENERATION_BUFFER_SECONDS);
    setTimeout(() => generateStep(loopId), delta * 1000);
}
  
/**
 * Decode the output index and play it on the piano.
 */
function playOutput(index: number) {
    let offset = 0;
    for (const eventRange of EVENT_RANGES) {
        const eventType = eventRange[0] as string;
        const minValue = eventRange[1] as number;
        const maxValue = eventRange[2] as number;
        if (offset <= index && index <= offset + maxValue - minValue) {
            if (eventType === 'note_on') {
            const noteNum = index - offset;
            activeNotes.set(noteNum, currentPianoTimeSec);
            return piano.keyDown(
                noteNum, currentPianoTimeSec, currentVelocity * controlGain / 100);
            } else if (eventType === 'note_off') {
                const noteNum = index - offset;
  
                const activeNoteEndTimeSec = activeNotes.get(noteNum);
                // If the note off event is generated for a note that hasn't been
                // pressed, just ignore it.
                if (activeNoteEndTimeSec == null) {
                    return;
                }
                const timeSec =
                    Math.max(currentPianoTimeSec, activeNoteEndTimeSec + .5);
                piano.keyUp(noteNum, timeSec);
                activeNotes.delete(noteNum);
                return;
            } else if (eventType === 'time_shift') {
                currentPianoTimeSec += (index - offset + 1) / STEPS_PER_SECOND;
                activeNotes.forEach((timeSec, noteNum) => {
                    if (currentPianoTimeSec - timeSec > MAX_NOTE_DURATION_SECONDS) {
                        console.info(
                        `Note ${noteNum} has been active for ${
                        currentPianoTimeSec - timeSec}, ` +
                        `seconds which is over ${MAX_NOTE_DURATION_SECONDS}, will ` +
                        `release.`);
                        piano.keyUp(noteNum, currentPianoTimeSec);
                        activeNotes.delete(noteNum);
                    }
                });
            return currentPianoTimeSec;
            } else if (eventType === 'velocity_change') {
                currentVelocity = (index - offset + 1) * Math.ceil(127 / VELOCITY_BINS);
                currentVelocity = currentVelocity / 127;
                return currentVelocity;
            } else {
                throw new Error('Could not decode eventType: ' + eventType);
            }
        }
        offset += maxValue - minValue + 1;
    }
    throw new Error(`Could not decode index: ${index}`);
}
  
// Reset the RNN repeatedly so it doesn't trail off into incoherent musical
// babble.
function resetRnnRepeatedly() {
    if (modelReady) {
        resetRnn();
        generateStep(currentLoopId);
    }
    setTimeout(() => {
    }, 1000);
    setTimeout(resetRnnRepeatedly, RESET_RNN_FREQUENCY_MS);
}
setTimeout(resetRnnRepeatedly, RESET_RNN_FREQUENCY_MS);
