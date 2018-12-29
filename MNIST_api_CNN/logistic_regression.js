require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');
const mnist = require('mnist-data');


loadData = () => {
    const mnistData = mnist.training(0,10000);

    const features = tf.stack(mnistData.images.values.map(image => tf.tensor(image).div(255).squeeze().expandDims(-1)));
    console.log("features",features.shape)

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
const testFeatures = tf.stack(testMnistData.images.values.map(image => tf.tensor(image).div(255).squeeze().expandDims(-1)));
console.log("testFeatures",testFeatures.shape)
const testEncodedLabels = testMnistData.labels.values.map(
    label => {
        const row = new Array(10).fill(0);
        row[label] = 1;
        return row;

    }
)

const xs = features
const ys = tf.tensor(labels);
const xt = testFeatures
const yt = tf.tensor(testEncodedLabels);

// Build and compile model.
const model = tf.sequential();
model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 3,
    filters: 16,
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
  model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
  model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
  model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
  model.add(tf.layers.flatten({}));
  model.add(tf.layers.dense({units: 64, activation: 'relu'}));
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
  model.compile({optimizer: 'rmsprop', loss: 'categoricalCrossentropy',  metrics: ['accuracy']});

// Train model with fit().
train = async () => {
      for (let i = 1; i < 10; ++i) {
       const z = await model.fit(xs, ys, {batchSize: 10, epochs: 10});
       console.log("Loss after Epoch " + i + " : " + z.history.loss[0], "| ","Accuracy: ",z.history.acc[0]);
      }
  }

//Test
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


//const xSinglePred = ([[]]);

SinglePredictiction = () => {
    const sp = model.predict(xSinglePred);
    console.log("Single predictiction:",sp.argMax(1).get(0));
}

exec = async () => {
    await train();
    const accuracy = await test(xt, yt);
   console.log("Accuracy on testset:", accuracy)
    //SinglePredictiction();
  }

exec();

