require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const lreg = require('./lreg');
const plot = require('node-remote-plot');
const _ = require('lodash');

let {features, labels, testFeatures, testLabels} = loadCSV('cars.csv', 
        {
            shuffle: true,
            splitTest: 50,
            dataColumns: ['horsepower', 'displacement','weight'],
            labelColumns: ['mpg'],
            converters: {
                mpg: value => {
                    const mpg = parseFloat(value);

                    if (mpg < 15) {
                        return [1,0,0];
                    } else if (mpg < 30) {
                        return [0,1,0];
                    } else {
                        return [0,0,1];
                    }
        }
    }
});

const reg = new lreg(features,_.flatMap(labels), {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 10

});

reg.train();

console.log("Accuracy on test set: ", reg.test(testFeatures,_.flatMap(testLabels)));

reg.predict([[100,200,2.223]]).print();
 