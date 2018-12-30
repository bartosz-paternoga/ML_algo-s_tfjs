require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const _ = require('lodash');

let {features, labels, testFeatures, testLabels} = loadCSV('titanicData.csv', 
        {
            shuffle: true,
            splitTest: 100,
            dataColumns: ['pclass', 'sex', 'sibsp', 'parch', 'cabin', 'embarked', 'age', 'fare'],
            labelColumns: ['survived'],
            converters: {
                sex: value => {
                    const sex = value;

                    if (sex === "male") {
                        return [1,0];
                    } else if (sex === "female") {
                        return [0,1];
                    }
                },
                pclass: value => {
                    const pclass = parseFloat(value);

                    if (pclass === 1) {
                        return [1,0,0];
                    } else if (pclass === 2) {
                        return [0,1,0];
                    } else {
                        return [0,0,1];
                    }
                },
                cabin: value => {
                    const cabin = value;

                    if (cabin === "y") {
                        return [1,0];
                    } else {
                        return [0,1];
                    }
                },
                embarked: value => {
                    const embarked = value;

                    if (embarked === "C") {
                        return [1,0,0];
                    } else if (embarked === "S") {
                        return [0,1,0];
                    } else {
                        return [0,0,1];
                    }
                }

    }
});

let F1 = [];
for (let i = 0; i < features.length; i++) { 
     const arr1 = features[i].slice(0,6);
    F1.push(arr1);
    };

const F1xs = tf.tensor((_.flatMap(_.flatMap(F1)))).reshape([features.length,12]);

let F2 = [];
for (let i = 0; i < features.length; i++) { 
     const arr1 = features[i].slice(6,8);
    F2.push(arr1);
    };


let F1t = [];
for (let i = 0; i < testFeatures.length; i++) { 
    const arr1 = testFeatures[i].slice(0,6);
    F1t.push(arr1);
    };
    
const F1xt = tf.tensor((_.flatMap(_.flatMap(F1t)))).reshape([testFeatures.length,12]);
    
let F2t = [];
for (let i = 0; i < testFeatures.length; i++) { 
    const arr1 = testFeatures[i].slice(6,8);
    F2t.push(arr1);
    };


const {mean, variance} = tf.moments(F2,0);

standarize = (inpt) => {
    inpt = tf.tensor(inpt);
   
    return inpt.sub(mean).div(variance.pow(0.5)); 
  };


const xs = F1xs.concat(standarize(F2),1);
console.log(xs.shape)
//const xs = tf.tensor((_.flatMap(_.flatMap(features)))).reshape([features.length,14])
const ys = tf.tensor(_.flatMap(labels));

const xt = F1xt.concat(standarize(F2t),1);
console.log(xt.print())
//const xt = tf.tensor((_.flatMap(_.flatMap(testFeatures)))).reshape([testFeatures.length,14]);
const yt = tf.tensor(_.flatMap(testLabels));



// Build and compile model.
const model = tf.sequential();
model.add(tf.layers.dense({units: 64, inputShape: [14], activation: 'relu'}));
model.add(tf.layers.dense({units: 64, activation: 'relu'}));
model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
model.compile({optimizer: 'adam',loss: tf.losses.logLoss});

// Train model with fit().
train = async () => {
      for (let i = 1; i < 20; ++i) {
       const z = await model.fit(xs, ys, {batchSize: 10, epochs: 10});
       console.log("Loss after Epoch " + i + " : " + z.history.loss[0]);
      }
  };


test = async (textX,testY) => {

    const predictictions =  model.predict(xt).round().squeeze();
    console.log("predictictions:",predictictions.print());

    testLabels = yt;
    console.log("testLabels:",testLabels.print());

    const incorrect = predictictions
        .notEqual(testLabels)
        .sum()
        .get();

        const accuracy =  (predictictions.shape[0] - incorrect) / predictictions.shape[0]; 

        console.log("Accuracy on test set:", accuracy);
};


//const xSinglePred = standarize(([[100,200,2.23]]));

SinglePredictiction = () => {
    const sp = model.predict(xSinglePred);
    console.log("Single predictiction:",sp.argMax(1).get(0));
};

exec = async () => {
    await train();
 await test();
 //   SinglePredictiction();
  };

exec();

