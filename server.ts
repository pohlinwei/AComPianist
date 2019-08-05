import express = require('express');
import child_process = require('child_process');

const app: express.Application = express();
const NO_CHILD_PROCESS: number = -1;
// process identifier (PID) of child process 
let current_pid: number = NO_CHILD_PROCESS;

app.use(express.static(__dirname));

const bodyParser = require('body-parser');
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.post('/get_params', (req, res) => {
    // check if there are any incomplete child processes
    if (current_pid != NO_CHILD_PROCESS) {
        // kill any previous processes
        const spawn = child_process.spawn;
        spawn('kill', [String(current_pid)]);
        console.log('Previous preprocess (' + current_pid + ') killed.');
    }

    // obtain parameters for music generation
    // retrieve data url of image
    const image: string = req.body.image;
    const spawn = child_process.spawn;
    const get_params = spawn('python3', ['main.py', image]);
    // update PID for new get_params process
    current_pid = get_params.pid;
    console.log('Server received image and new process is starting. New pid: ' + current_pid);
    let parameters = '';
    let count = 0;
    get_params.stdout.on('data', (data) => {
        parameters = parameters + data.toString();
        console.log('Parameters for this image: ' + parameters);
        count++;
    })
    get_params.stderr.on('err', (err) => {
        console.error(err);
    })
    get_params.on('close', () => {
        // indicate that get_params completed successfully
        current_pid = NO_CHILD_PROCESS;
        // send parameters for music generation
        res.end(parameters);
    });
})

app.listen(3000, function() {
    console.log("Go to localhost:3000 to view app.");
})
