require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
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

const {mean, variance} = tf.moments(features,0);

standarize = (inpt) => {
    inpt = tf.tensor(inpt);
   
    return inpt.sub(mean).div(variance.pow(0.5)); 
  }

const xs = standarize(features);
const ys = tf.tensor(_.flatMap(labels));
const xt = standarize(testFeatures)
const yt = tf.tensor(_.flatMap(testLabels));


// Build and compile model.
const model = tf.sequential();
model.add(tf.layers.dense({units: 3, inputShape: [3], activation: 'softmax'}));
// model.add(tf.layers.dense({units: 32, activation: 'relu'}));
// model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
model.compile({optimizer: 'sgd', loss: 'categoricalCrossentropy'});

// Train model with fit().
train = async () => {
      for (let i = 1; i < 10; ++i) {
       const z = await model.fit(xs, ys, {batchSize: 10, epochs: 10});
       console.log("Loss after Epoch " + i + " : " + z.history.loss[0]);
      }
  }


test = async (textX,testY) => {

    const predictictions =  model.predict(xt).argMax(1)
    console.log("predictictions:",predictictions.print())

    testLabels = yt.argMax(1);
    console.log("testLabels:",testLabels.print())

    const incorrect = predictictions
        .notEqual(testLabels)
        .sum()
        .get();

        const accuracy =  (predictictions.shape[0] - incorrect) / predictictions.shape[0]; 

        console.log("Accuracy on test set:", accuracy);
}  


const xSinglePred = standarize(([[100,200,2.23]]));

SinglePredictiction = () => {
    const sp = model.predict(xSinglePred);
    console.log("Single predictiction:",sp.argMax(1).get(0));
}

exec = async () => {
    await train();
    await test();
    SinglePredictiction();
  }

exec();

