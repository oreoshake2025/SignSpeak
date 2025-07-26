const videoElement = document.getElementById('webcam');
const letterElement = document.getElementById('letter');

function dummyClassifier(landmarks) {
  // You can replace this with real classification later
  if (!landmarks) return '?';

  // Example: classify based on position of thumb tip (landmarks[4])
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const distance = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);

  if (distance < 0.05) return 'A';
  if (distance < 0.1) return 'B';
  return 'C';
}

function onResults(results) {
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];
    const letter = dummyClassifier(landmarks);
    letterElement.innerText = letter;
  } else {
    letterElement.innerText = '...';
  }
  console.log("Landmarks:", results.multiHandLandmarks);

}

// Init MediaPipe
const hands = new Hands({
  locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 0,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

hands.onResults(onResults);

// Setup Camera
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({ image: videoElement });
  },
  width: 640,
  height: 480
});
camera.start();
