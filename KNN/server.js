const express = require('express');
const app = express();
const path = require('path');
app.use(express.static('client/build'));
require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const _ = require('lodash');

process.chdir('./client/public');

let que = [];


app.param('id', async (req, res, next, id) => {

	try {

		function knn (features, labels, predictionPoint, k) {
			debugger;
			const { mean, variance } = tf.moments(features, 0);
			const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5))

			return features
				.sub(mean)
				.div(variance.pow(0.5))
				.sub(scaledPrediction)
				.pow(2)
				.sum(1)
				.pow(0.5)
				.expandDims(1)
				.concat(labels,1)
				.unstack()
				.sort((a,b) => a.get(0) > b.get(0) ? 1 : -1)
				.slice (0, k)
				.reduce ((acc, pair) => acc + pair.get(1), 0) / k;
		}



		let {features, labels} = loadCSV('weight-height.csv', 
			{
			shuffle: true,
			dataColumns: ['Height', 'Weight'],
			labelColumns: ['Gender1']
			}
		);

		features = tf.tensor(features);
		labels = tf.tensor(labels);

		const modified = id;
		const idx = id.split("-");
		console.log( "idx:", idx); 

		que.push(idx);
		const num = que[que.length-1];
		console.log(' [x] Asking (%s)', num, tf.tensor([num][0]).print());

		const v = await knn(features, labels , tf.tensor([num][0]), 10);

		if (v >= 0.5) {
			z = "Male"
		} else {
			z = "Female"
		};

		console.log( "Result:", v, z); 

		res.send(z);

	} catch (e) {
			next(e);
		}
});


app.get('/api/user/:id',(req,res)=>{});

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname+'/client/build/index.html'));
});

const PORT = process.env.PORT || 3001;

app.listen(PORT,function () {
  console.log('Ready');
});



