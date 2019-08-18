const tf = require('@tensorflow/tfjs-node');

let model;  // global variable to store the neural net

// returns the index of the array's maximum value
function maxIndex(array) {
    let maxIndex = 0;
    for (let i = 1; i < array.length; i++) {
        if (array[i] > array[maxIndex]) {
            maxIndex = i;
        }
    }
    return maxIndex;
}

// defines the architecture of the neural net
function defineModel() {
    model = tf.sequential();
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [10] }));
    model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 30, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 20, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy' });
}

// generates new training and testing data
function generateData() {
    let inputs = [];
    let outputs = [];

    for (let i = 0; i < 10000; i++) {
        let tempIn = [];
        let tempOut = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        for (let j = 0; j < 10; j++) {
            tempIn.push(Math.random());
        }

        tempOut[maxIndex(tempIn)] = 1;

        inputs.push(tempIn);
        outputs.push(tempOut);
    }

    let testInputs = [];
    let testOutputs = [];

    for (let i = 0; i < 1000; i++) {
        let tempIn = [];
        let tempOut = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        for (let j = 0; j < 10; j++) {
            tempIn.push(Math.random());
        }

        tempOut[maxIndex(tempIn)] = 1;

        testInputs.push(tempIn);
        testOutputs.push(tempOut);
    }

    const input = tf.tensor2d(inputs);
    const output = tf.tensor2d(outputs);
    const testInput = tf.tensor2d(inputs);
    const testOutput = tf.tensor2d(outputs);

    return { 
        train: {
            input: input, 
            output: output
        },
        test: {
            input: testInput,
            output: testOutput
        }
    };
}

// starts training process
async function trainModel(data, epochs) {
    await model.fit(data.input, data.output, {
        epochs: epochs,
        callbacks: {
            onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
        }
    });
    try {
        await model.save('file://' + __dirname + '/saved/maximum');
        console.log('model saved!');
    } catch (e) {
        console.log(e);
    }
}

// tests the model using the test-dataset
async function test(testData) {
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
    const data = generateData();
    defineModel();
    await trainModel(data.train, 50);
    await test(data.test);
}

// start
main();
