'use strict'

const ml = require('machine_learning');

let x = [[1,1,1,0,0,0],
         [1,0,1,0,0,0],
         [1,1,1,0,0,0],
         [0,0,1,1,1,0],
         [0,0,1,1,0,0],
         [0,0,1,1,1,0]];

let y = [[1, 0],
         [1, 0],
         [1, 0],
         [0, 1],
         [0, 1],
         [0, 1]];

let classifier = new ml.LogisticRegression({
    'input' : x,
    'label' : y,
    'n_in' : 6,
    'n_out' : 2
});

classifier.set('log level',1);

let training_epochs = 800, lr = 0.01;

classifier.train({
    'lr' : lr,
    'epochs' : training_epochs
});

x = [[1, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 0],
     [1, 1, 1, 1, 1, 0]];

console.log("Result : ",classifier.predict(x));


console.log("Entropy : "+classifier.getReconstructionCrossEntropy());