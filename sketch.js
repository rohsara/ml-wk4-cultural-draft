let model, video, keypoints, predictions=[]; 
let okL, okR, fcL, fcR, roL, roR, tuL, tuR;
// Create a KNN classifier
const classifier = knnClassifier.create();

const LABELS_MAP = {
	'Ok': 0,
	'FingersCrossed': 1,
	'RockOn': 2,
	'ThumbsUp': 3 
};

function preload() {
	video = createCapture(VIDEO, () => {
	  loadHandTrackingModel();
	});
	// video.size(480, 360);
    video.hide();
    
	// Create the UI buttons
	createButtons();
    okL = loadImage('assets/okClassifyL.png');
    okR = loadImage('assets/okClassifyR.png');
    fcL = loadImage('assets/fcClassifyL.png');
    fcR = loadImage('assets/fcClassifyR.png');
    roL = loadImage('assets/roClassifyL.png');
    roR = loadImage('assets/roClassifyR.png');
    tuL = loadImage('assets/tuClassifyL.png');
    tuR = loadImage('assets/tuClassifyR.png');
}

function setup() {
	const canvas = createCanvas(480, 360);
	canvas.parent('canvasContainer');
}

async function loadHandTrackingModel() {
	// Load the MediaPipe handpose model.
	model = await handpose.load();
	select('#status').html('Hand Tracking Model Loaded')
	predictHand();
}

function draw() {
	background(255);
	if (model) image(video, 0, 0);
    filter(GRAY);
	if (predictions.length > 0) {
	  // We can call both functions to draw all keypoints and the skeletons
	  drawKeypoints();
	  drawSkeleton();
	}
}

async function predictHand() {
	// Pass in a video stream (or an image, canvas, or 3D tensor) to obtain a
	// hand prediction from the MediaPipe graph.
	predictions = await model.estimateHands(video.elt);
  
	setTimeout(() => predictHand(), 200);
}

// Add the current hand tracking data to the classifier
function addExample(label) {
	if (predictions.length > 0) {
	  const features = predictions[0].landmarks;
	  const tensors = tf.tensor(features)
	  // Add an example with a label to the classifier
	  classifier.addExample(tensors, label);
	  updateCounts();
	} else {
	  console.log('No gesture is detected')
	}
}

// Predict the current frame.
async function classify() {
	// Get the total number of labels from classifier
	const numLabels = classifier.getNumClasses();
	if (numLabels <= 0) {
	  console.error('There is no examples in any label');
	  return;
	}
	if (predictions.length > 0) {
		const results = await classifier.predictClass(tf.tensor(predictions[0].landmarks));
		if (results.confidences) {
			const confidences = results.confidences;
			
			// result.label is the label that has the highest confidence
			if (results.label) {
				select('#result').html(results.label);
				select('#confidence').html(`${confidences[results.label] * 100} %`);
                if(results.label === 'Ok') {
                    document.getElementById("left").src = "/assets/okClassifyL.png";
                    document.getElementById("right").src = "/assets/okClassifyR.png";
                }
                if(results.label === 'FingersCrossed') {
                    document.getElementById("left").src = "/assets/fcClassifyL.png";
                    document.getElementById("right").src = "/assets/fcClassifyR.png";
                }
                if(results.label === 'RockOn') {
                    document.getElementById("left").src = "/assets/roClassifyL.png";
                    document.getElementById("right").src = "/assets/roClassifyR.png";
                }
                if(results.label === 'ThumbsUp') {
                    document.getElementById("left").src = "/assets/tuClassifyL.png";
                    document.getElementById("right").src = "/assets/tuClassifyR.png";
                }

			}

			select('#confidenceOk').html(`${confidences['Ok'] ? confidences['Ok'] * 100 : 0} %`);
			select('#confidenceFingersCrossed').html(`${confidences['FingersCrossed'] ? confidences['FingersCrossed'] * 100 : 0} %`);
			select('#confidenceRockOn').html(`${confidences['RockOn'] ? confidences['RockOn'] * 100 : 0} %`);
			select('#confidenceThumbsUp').html(`${confidences['ThumbsUp'] ? confidences['ThumbsUp'] * 100 : 0} %`);
		}
		classify();
  	} else {
    	setTimeout(() => classify(), 1000);
  	}
}

// Update the example count for each label	
function updateCounts() {
	const counts = classifier.getClassExampleCount();

	select('#exampleOk').html(counts['Ok'] || 0);
	select('#exampleFingersCrossed').html(counts['FingersCrossed'] || 0);
	select('#exampleRockOn').html(counts['RockOn'] || 0);
	select('#exampleThumbsUp').html(counts['ThumbsUp'] || 0);
}

// Clear the examples in one label
function clearLabel(label) {
	classifier.clearClass(label);
	updateCounts();
}
  
// Clear all the examples in all labels
function clearAllLabels() {
	classifier.clearAllClasses();
	updateCounts();
}
  
// A util function to create UI buttons
function createButtons() {
	// When the A button is pressed, add the current frame
	// from the video with a label of "do" to the classifier
    buttonA = select('#addClassOk');
    buttonA.mousePressed(function(){
        addExample('Ok');
    }); 

    buttonB = select('#addClassFingersCrossed');
    buttonB.mousePressed(function(){
        addExample('FingersCrossed');
    }); 

	buttonC = select('#addClassRockOn');
	buttonC.mousePressed(function(){
		addExample('RockOn');
	}); 
	
	buttonD = select('#addClassThumbsUp');
	buttonD.mousePressed(function(){
		addExample('ThumbsUp');
	}); 
	
	// Reset Buttons
    resetBtnA = select('#resetOk');
    resetBtnA.mousePressed(function(){
        addExample('Ok');
    }); 

    resetBtnB = select('#resetFingersCrossed');
    resetBtnB.mousePressed(function(){
        addExample('FingersCrossed');
    }); 

	resetBtnC = select('#resetRockOn');
	resetBtnC.mousePressed(function(){
		addExample('RockOn');
	}); 
	
	resetBtnD = select('#resetThumbsUp');
	resetBtnD.mousePressed(function(){
		addExample('ThumbsUp');
	}); 

	// Predict Button
	buttonPredict = select('#buttonPredict');
	buttonPredict.mousePressed(classify);
	
	// Clear all classes button
	buttonClearAll = select('#clearAll');
	buttonClearAll.mousePressed(clearAllLabels);
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints()  {
	let prediction = predictions[0];
	for (let j = 0; j < prediction.landmarks.length; j++) {
	  let keypoint = prediction.landmarks[j];
	  fill(255, 0, 0);
	  noStroke();
	  ellipse(keypoint[0], keypoint[1], 10, 10);
	}
}
  
  // A function to draw the skeletons
function drawSkeleton() {
	let annotations = predictions[0].annotations;
	stroke(255, 0, 0);
	for (let j = 0; j < annotations.thumb.length - 1; j++) {
	  line(annotations.thumb[j][0], annotations.thumb[j][1], annotations.thumb[j + 1][0], annotations.thumb[j + 1][1]);
	}
	for (let j = 0; j < annotations.indexFinger.length - 1; j++) {
	  line(annotations.indexFinger[j][0], annotations.indexFinger[j][1], annotations.indexFinger[j + 1][0], annotations.indexFinger[j + 1][1]);
	}
	for (let j = 0; j < annotations.middleFinger.length - 1; j++) {
	  line(annotations.middleFinger[j][0], annotations.middleFinger[j][1], annotations.middleFinger[j + 1][0], annotations.middleFinger[j + 1][1]);
	}
	for (let j = 0; j < annotations.ringFinger.length - 1; j++) {
	  line(annotations.ringFinger[j][0], annotations.ringFinger[j][1], annotations.ringFinger[j + 1][0], annotations.ringFinger[j + 1][1]);
	}
	for (let j = 0; j < annotations.pinky.length - 1; j++) {
	  line(annotations.pinky[j][0], annotations.pinky[j][1], annotations.pinky[j + 1][0], annotations.pinky[j + 1][1]);
	}
  
	line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.thumb[0][0], annotations.thumb[0][1]);
	line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.indexFinger[0][0], annotations.indexFinger[0][1]);
	line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.middleFinger[0][0], annotations.middleFinger[0][1]);
	line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.ringFinger[0][0], annotations.ringFinger[0][1]);
	line(annotations.palmBase[0][0], annotations.palmBase[0][1], annotations.pinky[0][0], annotations.pinky[0][1]);
}  