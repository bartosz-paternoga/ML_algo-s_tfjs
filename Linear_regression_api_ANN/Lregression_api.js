require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

let {features, labels, testFeatures, testLabels} = loadCSV('cars.csv', 
        {
            shuffle: true,
            splitTest: 50,
            dataColumns: ['horsepower', 'weight', 'displacement'],
            labelColumns: ['mpg']
        }
    );

const {mean, variance} = tf.moments(features,0);

standarize = (inpt) => {
    inpt = tf.tensor(inpt);
   
    return inpt.sub(mean).div(variance.pow(0.5)); 
  }

const xs = standarize(features);
const ys = tf.tensor(labels);
const xt = standarize(testFeatures)
const yt = tf.tensor(testLabels);

// Build and compile model.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [3]}));
// model.add(tf.layers.dense({units: 1, inputShape: [3]}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

// Train model with fit().
train = async () => {
      for (let i = 1; i < 10 ; ++i) {
       const z = await model.fit(xs, ys, {batchSize: 10, epochs: 100});
       console.log("Loss after Epoch " + i + " : " + z.history.loss[0]);
      }
  }



test = (xt,yt) => {

    const predictictions =  model.predict(xt)
    console.log("predictictions",predictictions.print())

    const res = yt
        .sub(predictictions)
        .pow(2)
        .sum()
        .get();
    const tot = yt
        .sub(yt.mean())
        .pow(2)
        .sum()
        .get();

        return 1 - res/tot
}  



a = async () => {
    await train();
    // Run inference with predict().
    const r2 = test(xt,yt);
    console.log('R2 is', r2);
  }

a();
