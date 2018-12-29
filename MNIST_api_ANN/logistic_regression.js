require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const lreg = require('./logistic_regression');
const plot = require('node-remote-plot');
const mnist = require('mnist-data');
const _ = require('lodash');

loadData = () => {
    const mnistData = mnist.training(0,20000);

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



const testMnistData = mnist.testing(0,5000);
const testFeatures = testMnistData.images.values.map(image => _.flatMap(image));

const testEncodedLabels = testMnistData.labels.values.map(
    label => {
        const row = new Array(10).fill(0);
        row[label] = 1;
        return row;

    }
)


const {mean, variance} = tf.moments(features,0);

  standarize = (inpt) => {
    inpt = tf.tensor(inpt);
    const filler = variance.cast("bool").logicalNot().cast("float32");
    this.mean = mean;
    this.variance = variance.add(filler);

    return inpt.sub(mean).div(this.variance.pow(0.5)); 
}

const xs = standarize(features);

const ys = tf.tensor(labels);

const xt = standarize(testFeatures)
const yt = tf.tensor(testEncodedLabels);


// Build and compile model.
const model = tf.sequential();
model.add(tf.layers.dense({units: 10, inputShape: [784], activation: 'softmax'}));
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


test = async (xt,yt) => {

        const predictictions =  model.predict(xt).argMax(1)
        console.log("predictictions:",predictictions.print())
        
        testLabels = yt.argMax(1);
        console.log("testLabels:",testLabels.print())

        const incorrect = predictictions
            .notEqual(testLabels)
            .sum()
            .get();

        return (predictictions.shape[0] - incorrect) / predictictions.shape[0]; 
    }


//const xSinglePred = standarize(([[]]));

SinglePredictiction = () => {
    const sp = model.predict(xSinglePred);
    console.log("Single predictiction:",sp.argMax(1).get(0));
}

exec = async () => {
    await train();
    const accuracy = await test(xt, yt);
    console.log("Accuracy:", accuracy)
    //SinglePredictiction();
  }

exec();

