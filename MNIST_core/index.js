require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const lreg = require('./logistic_regression');
const plot = require('node-remote-plot');
const mnist = require('mnist-data');
const _ = require('lodash');

loadData = () => {
    const mnistData = mnist.training(0,25000);

    const features = mnistData.images.values.map(image => _.flatMap(image));

    const encodedLabels = mnistData.labels.values.map(
        label => {
            const row = new Array(10).fill(0);
            row[label] = 1;
            return row;
        }
    );
    return {features,labels: encodedLabels}
}

const {features,labels} = loadData();

const regression = new lreg(features, labels,{
    learningRate: 1,
    iterations: 50,
    batchSize: 100

})

regression.train();

const testMnistData = mnist.testing(0,5000);
const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));

const testEncodedLabels = testMnistData.labels.values.map(
    label => {
        const row = new Array(10).fill(0);
        row[label] = 1;
        return row;

    }
)

const accuracy = regression.test(testFeatures,testEncodedLabels);
console.log("Accuracy:", accuracy)
