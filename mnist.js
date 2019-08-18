const tf = require('@tensorflow/tfjs-node');
var fs = require('fs');

let model; // global variable to store the neural net

// defines the architecture of the neural net
function defineModel() {
    model = tf.sequential();
    model.add(tf.layers.dense({ units: 100, activation: 'sigmoid', inputShape: [784] }));
    model.add(tf.layers.dense({ units: 100, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 20, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 20, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 20, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 20, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 20, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 20, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 20, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    const optimizer = tf.train.adam();
    model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
}

// loads the MNIST data from harddrive and reshapes it into tensors
function openMNIST() {
    const tkI = fs.readFileSync('MNIST/t10k-images.idx3-ubyte');
    const tkL = fs.readFileSync('MNIST/t10k-labels.idx1-ubyte');
    const trainI = fs.readFileSync('MNIST/train-images.idx3-ubyte');
    const trainL = fs.readFileSync('MNIST/train-labels.idx1-ubyte');

    const testLabels = [...tkL].slice(8);
    const testImages = [...tkI].slice(16);
    const trainLabels = [...trainL].slice(8);
    const trainImages = [...trainI].slice(16);

    const trainInput = tf.tensor2d(trainImages, [60000, 784]);
    const trainOutput = tf.oneHot(trainLabels, 10);
    const testInput = tf.tensor2d(testImages, [10000, 784]);
    const testOutput = tf.oneHot(testLabels, 10);

    return { 
        train: { 
            input: trainInput, 
            output: trainOutput 
        }, 
        test: { 
            input: testInput, 
            output: testOutput 
        } 
    };
}

// starts the training process
async function trainModel(data, epochs) {
    await model.fit(data.input, data.output, {
        epochs: epochs,
        callbacks: {
            onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
        }
    });
    try {
        await model.save('file://' + __dirname + '/saved/mnist');
        console.log('model saved!');
    } catch (e) {
        console.log(e);
    }
}

// tests the model using the test-dataset
async function testMNIST(testData) {
    const result = await model.predict(testData.input).argMax(1).array();
    const output = await testData.output.argMax(1).array();
    let errorCount = 0;
    for (let i = 0; i < output.length; i++) {
        if (output[i] !== result[i]) {
            errorCount += 1;
        }
    }
    console.log(errorCount + ' errors, ' + (errorCount * 100 / output.length) + '% error rate');
}

// main function to run the script asynchronously
async function main() {
    const mnist = openMNIST();
    defineModel();
    await trainModel(mnist.train, 100);
    await testMNIST(mnist.test);
}

// start
main();
