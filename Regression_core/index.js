require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const lreg = require('./lreg');
const plot = require('node-remote-plot');


let {features, labels, testFeatures, testLabels} = loadCSV('cars.csv', 
        {
            shuffle: true,
            splitTest: 50,
            dataColumns: ['horsepower', 'weight', 'displacement'],
            labelColumns: ['mpg']
        }
    );

const reg = new lreg(features,labels, {
    learningRate: 0.1,
    iterations: 3,
    batchSize: 10

});

reg.train();

const r2 = reg.test(testFeatures,testLabels);
console.log('R2 is', r2);
// console.log('mse hist', reg.mseHistory);

// plot ({
//     // x: reg.bHistory,
//     x: reg.mseHistory.reverse(),
//     xlabel: 'iterations #',
//     ylabel: 'MSE'
// });


reg.predict([[120,2,380],
    [130,2.5,330]]).print();
console.log('weights', reg.weights.print());