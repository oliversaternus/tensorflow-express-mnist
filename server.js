var express = require('express');
var tf = require('@tensorflow/tfjs-node');
var bodyParser = require('body-parser');

var model; // global variable to store the pretrained neural net
var app;   // global variable to store the server

/*-------------HELPER FUNCTIONS----------------*/

// returns the recognized number
async function predict(tensor) {
    const result = await model.predict(tensor).argMax(1).array();
    return result[0];
}

// loads the pretrained model
async function loadModel() {
    model = await tf.loadLayersModel('file://' + __dirname + '/saved/convolutional/model.json');
}

// reshapes the transmitted 28x28 array to tensor
function tensor(input) {
    let flat = [];
    for (let i = 0; i < 28; i++) {
        flat = flat.concat(input[i]);
    }
    const tensor = tf.tensor(flat, [1, 28, 28, 1]);
    return tensor;
}

/*----------------- INITIALIZING SERVER ------------------*/

app = express();
app.use(bodyParser.json());

// default route returns the html document
app.get('/', function (req, res) {
    res.sendFile(__dirname + '/canvas.html');
});

// returns the prediction for a sent 28x28 number array, representing an image
app.post('/api/predict', async function (req, res) {
    try {
        const image = req.body;
        const prediction = await predict(tensor(image));
        res.status(200).send({ prediction: prediction });
    } catch (e) {
        res.sendStatus(500);
    }
});

/*---------------------------------------------------------------*/

// main function to start the server asynchronously
async function main() {
    await loadModel();
    app.listen(3000, function () {
        console.log('App listening on port 3000!');
    });
}

// start
main();