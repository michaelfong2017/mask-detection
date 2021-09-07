import React, { useEffect, useRef } from "react";
import ReactDOM from "react-dom";
import {
  HashRouter,
  Switch,
  Route,
} from "react-router-dom";

import * as tf from '@tensorflow/tfjs';
// import {loadGraphModel} from '@tensorflow/tfjs-converter';
import "./styles.css";
tf.setBackend('webgl');

async function load_model() {
  // It's possible to load the model locally or from a repo
  // You can choose whatever IP and PORT you want in the "http://127.0.0.1:8080/model.json" just set it before in your https server
  //const model = await loadGraphModel("http://127.0.0.1:8080/model.json");
  const model = await tf.loadLayersModel("https://raw.githubusercontent.com/AIZOOTech/mask-detection-web-demo/master/tfjs-models/model.json");
  return model;
}

const id2class = { 0: "有口罩", 1: "無口罩" }

function decodeBBox(anchors, rawOutput, variances = [0.1, 0.1, 0.2, 0.2]) {
  const [anchorXmin, anchorYmin, anchorXmax, anchorYmax] = tf.split(anchors, [1, 1, 1, 1], -1);
  const anchorCX = tf.div(tf.add(anchorXmin, anchorXmax), 2);
  const anchorCY = tf.div(tf.add(anchorYmin, anchorYmax), 2);

  const anchorW = tf.sub(anchorXmax, anchorXmin);
  const anchorH = tf.sub(anchorYmax, anchorYmin);

  const rawOutputScale = tf.mul(rawOutput, tf.tensor(variances));
  const [rawOutputCX, rawOutputCY, rawOutputW, rawOutputH] = tf.split(rawOutputScale, [1, 1, 1, 1], -1);
  const predictCX = tf.add(tf.mul(rawOutputCX, anchorW), anchorCX);
  const predictCY = tf.add(tf.mul(rawOutputCY, anchorH), anchorCY);
  const predictW = tf.mul(tf.exp(rawOutputW), anchorW);
  const predictH = tf.mul(tf.exp(rawOutputH), anchorH);
  const predictXmin = tf.sub(predictCX, tf.div(predictW, 2));
  const predictYmin = tf.sub(predictCY, tf.div(predictH, 2));
  const predictXmax = tf.add(predictCX, tf.div(predictW, 2));
  const predictYmax = tf.add(predictCY, tf.div(predictH, 2));
  // eslint-disable-next-line
  const predictBBox = tf.concat([predictYmin, predictXmin, predictYmax, predictXmax], -1);
  return predictBBox
}

function anchorGenerator(featureMapSizes, anchorSizes, anchorRatios) {
  let anchorBBoxes = [];
  // eslint-disable-next-line
  featureMapSizes.map((featureSize, idx) => {
    const cx = tf.div(tf.add(tf.linspace(0, featureSize[0] - 1, featureSize[0]), 0.5), featureSize[0]);
    const cy = tf.div(tf.add(tf.linspace(0, featureSize[1] - 1, featureSize[1]), 0.5), featureSize[1]);
    const cxGrid = tf.matMul(tf.ones([featureSize[1], 1]), cx.reshape([1, featureSize[0]]));
    const cyGrid = tf.matMul(cy.reshape([featureSize[1], 1]), tf.ones([1, featureSize[0]]));
    // eslint-disable-next-line
    const cxGridExpend = tf.expandDims(cxGrid, -1);
    // eslint-disable-next-line
    const cyGridExpend = tf.expandDims(cyGrid, -1);
    // eslint-disable-next-line
    const center = tf.concat([cxGridExpend, cyGridExpend], -1);
    const numAnchors = anchorSizes[idx].length + anchorRatios[idx].length - 1;
    const centerTiled = tf.tile(center, [1, 1, 2 * numAnchors]);
    // eslint-disable-next-line
    let anchorWidthHeights = [];

    // eslint-disable-next-line
    for (const scale of anchorSizes[idx]) {
      const ratio = anchorRatios[idx][0];
      const width = scale * Math.sqrt(ratio);
      const height = scale / Math.sqrt(ratio);

      const halfWidth = width / 2;
      const halfHeight = height / 2;
      anchorWidthHeights.push(-halfWidth, -halfHeight, halfWidth, halfHeight);
      // width = tf.mul(scale, tf.sqrt(ratio));
      // height = tf.div(scale, tf.sqrt(ratio));

      // halfWidth = tf.div(width, 2);
      // halfHeight = tf.div(height, 2);
      // anchorWidthHeights.push(tf.neg(halfWidth), tf.neg(halfWidth), halfWidth, halfHeight);
    }

    // eslint-disable-next-line
    for (const ratio of anchorRatios[idx].slice(1)) {
      const scale = anchorSizes[idx][0];
      const width = scale * Math.sqrt(ratio);
      const height = scale / Math.sqrt(ratio);
      const halfWidth = width / 2;
      const halfHeight = height / 2;
      anchorWidthHeights.push(-halfWidth, -halfHeight, halfWidth, halfHeight);
    }
    const bboxCoord = tf.add(centerTiled, tf.tensor(anchorWidthHeights));
    const bboxCoordReshape = bboxCoord.reshape([-1, 4]);
    anchorBBoxes.push(bboxCoordReshape);
  })
  // eslint-disable-next-line
  anchorBBoxes = tf.concat(anchorBBoxes, 0);
  return anchorBBoxes;
}

let featureMapSizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]];
let anchorSizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]];
let anchorRatios = [[1, 0.62, 0.42], [1, 0.62, 0.42], [1, 0.62, 0.42], [1, 0.62, 0.42], [1, 0.62, 0.42]];

let anchors = anchorGenerator(featureMapSizes, anchorSizes, anchorRatios);

async function nonMaxSuppression(bboxes, confidences, confThresh, iouThresh, width, height, maxOutputSize = 100) {
  const bboxMaxFlag = tf.argMax(confidences, -1);
  const bboxConf = tf.max(confidences, -1);
  const keepIndices = await tf.image.nonMaxSuppressionAsync(bboxes, bboxConf, maxOutputSize, iouThresh, confThresh);
  // eslint-disable-next-line
  let results = []
  const keepIndicesData = keepIndices.dataSync();
  const bboxConfData = bboxConf.dataSync();
  const bboxMaxFlagData = bboxMaxFlag.dataSync();
  const bboxesData = bboxes.dataSync();
  // eslint-disable-next-line
  keepIndicesData.map((idx) => {
    const xmin = Math.round(Math.max(bboxesData[4 * idx + 1] * width, 0));
    const ymin = Math.round(Math.max(bboxesData[4 * idx + 0] * height, 0));
    const xmax = Math.round(Math.min(bboxesData[4 * idx + 3] * width, width))
    const ymax = Math.round(Math.min(bboxesData[4 * idx + 2] * height, height));
    results.push([[xmin, ymin, xmax, ymax],
    bboxMaxFlagData[idx], bboxConfData[idx]])
  });
  return results;
}

class SSD extends React.Component {
  videoRef = React.createRef();
  canvasRef = React.createRef();

  constructor(props) {
    super(props);
    this.state = {
      threshold: 0.75
    };
  }
  onThresholdChange(value) {
    this.setState({
      threshold: value
    });
  }

  componentDidMount() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webCamPromise = navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: "user",
            // width: { ideal: 260 }, 
            // height: { ideal: 260 } 
          }
        })
        .then(stream => {
          window.stream = stream;
          this.videoRef.current.srcObject = stream;
          return new Promise((resolve, reject) => {
            this.videoRef.current.onloadedmetadata = () => {
              resolve();
            };
          });
        });

      const modelPromise = load_model();

      Promise.all([modelPromise, webCamPromise])
        .then(values => {
          this.detectFrame(this.videoRef.current, values[0]);
        })
        .catch(error => {
          console.error(error);
        });
    }
  }

  detectFrame = async (video, model) => {
    tf.engine().startScope();
    const width = 600
    const height = 500
    const [rawBBoxes, rawConfidences] = model.predict(this.process_input(video))
    const bboxes = decodeBBox(anchors, tf.squeeze(rawBBoxes));
    const predictions = await nonMaxSuppression(bboxes, tf.squeeze(rawConfidences), 0.5, 0.5, width, height);
    this.renderPredictions(predictions, video);
    requestAnimationFrame(() => {
      this.detectFrame(video, model);
    });
    tf.engine().endScope();
  };

  process_input(video_frame) {
    let img = tf.browser.fromPixels(video_frame);
    img = tf.image.resizeBilinear(img, [260, 260]);
    img = img.expandDims(0).toFloat().div(tf.scalar(255));
    return img;
  };

  // buildDetectedObjects(scores, threshold, boxes, classes, classesDir) {
  //   const detectionObjects = []
  //   var video_frame = document.getElementById('frame');

  //   scores.forEach((score, i) => {
  //     if (score > threshold) {
  //       const bbox = [];
  //       const minY = boxes[i][0] * video_frame.offsetHeight;
  //       const minX = boxes[i][1] * video_frame.offsetWidth;
  //       const maxY = boxes[i][2] * video_frame.offsetHeight;
  //       const maxX = boxes[i][3] * video_frame.offsetWidth;
  //       bbox[0] = minX;
  //       bbox[1] = minY;
  //       bbox[2] = maxX - minX;
  //       bbox[3] = maxY - minY;
  //       detectionObjects.push({
  //         class: classes[i],
  //         label: classesDir[classes[i]].name,
  //         score: score.toFixed(4),
  //         bbox: bbox
  //       })
  //     }
  //   })
  //   return detectionObjects
  // }

  renderPredictions = predictions => {
    this.canvasRef.current.width = 600
    this.canvasRef.current.height = 500
    const ctx = this.canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    //Getting predictions
    // const boxes = predictions.map(prediction => prediction[0])
    // const classes = predictions.map(prediction => prediction[1])
    // const scores = predictions.map(prediction => prediction[2])

    // const detections = this.buildDetectedObjects(scores, threshold,
    //   boxes, classes, classesDir);

    for (var bboxInfo of predictions) {
      var bbox = bboxInfo[0];
      var classID = bboxInfo[1];
      var score = bboxInfo[2];

      if (score <= this.state.threshold) {
        continue
      }

      ctx.beginPath();
      ctx.lineWidth = "4";
      if (classID == 0) {
        ctx.strokeStyle = "green";
        ctx.fillStyle = "green";
      } else {
        ctx.strokeStyle = "red";
        ctx.fillStyle = "red";
      }

      ctx.rect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
      ctx.stroke();

      ctx.font = "30px Arial";

      let content = id2class[classID] + " " + score.toFixed(2);
      // ctx.fillText(content, bbox[0], bbox[1] < 60 ? (bbox[1] < 30 ? bbox[1] + 30 : bbox[1] + 10) : bbox[1] - 30);
      ctx.fillText(content, bbox[0] + 5, bbox[1] < 60 ? (bbox[1] < 30 ? bbox[1] + 8 : bbox[1] + 8) : bbox[1] + 8);
    }
  };

  render() {
    return (
      <div>
        <h1>Real-Time Face Mask Detection </h1>
        <h3>Single Shot MultiBox Detector (SSD). Model from <a href="https://github.com/AIZOOTech/mask-detection-web-demo">https://github.com/AIZOOTech/mask-detection-web-demo</a>.</h3>
        <br />
        <div>
          Threshold: <input type="text"
            className="form-control"
            value={this.state.threshold}
            onChange={e => this.onThresholdChange(e.target.value)} />
        </div>
        <video
          style={{ marginTop: '165px', height: '500px', width: "600px" }}
          className="size"
          autoPlay
          playsInline
          muted
          ref={this.videoRef}
          width="600"
          height="500"
          id="frame"
        />
        <canvas
          style={{ marginTop: '165px' }}
          className="size"
          ref={this.canvasRef}
          width="600"
          height="500"
        />
        <h3 style={{ marginTop: "500px" }}><a href="#">Faster R-CNN</a></h3>
      </div>
    );
  }
}

const App = () => {
  return (
    <HashRouter hashType="noslash">
      <Switch>
        <Route path="/ssd" component={SSD} />
        <Route path="/" component={FasterRCNN} />
      </Switch>

    </HashRouter>
  )
}

function useToggle(initialValue = true) {
  const [value, setValue] = React.useState(initialValue);
  const toggle = React.useCallback(() => {
    setValue(v => !v);
  }, []);
  return [value, toggle];
}
const FasterRCNN = () => {
  const [isOn, toggleIsOn] = useToggle()
  const videoRef = useRef(null)

  useEffect(() => {
    let lastTime = videoRef.current.currentTime
    videoRef.current.load()
    videoRef.current.currentTime = lastTime
    videoRef.current.play()
  }, [isOn])

  return (
    <div>
      <h1>Group M - Faster R-CNN Face Mask Detection </h1>
      <h3>A custom Faster R-CNN model is trained using custom data and configuration.</h3>
      <h3>To reproduce, please follow our tutorial on Github: <a target="_blank" href="https://github.com/JiaminWoo/Face_Mask_Detection">https://github.com/JiaminWoo/Face_Mask_Detection</a>.</h3>

      <br />

      <div style={{
        display: "flex",
        flexDirection: "row",
        alignItems: "flex-start"
      }}>
        <div style={{ marginLeft: "10px", fontSize: "1.5rem" }} >
          {isOn === true ? 'HD on' : 'HD off'}
        </div>
        <label className="switch" style={{ transform: "scale(0.7)" }}>
          <input type="checkbox" defaultChecked onClick={toggleIsOn} />
          <span className="slider round"></span>
        </label>
      </div>

      <div style={{ marginTop: "5px" }} />

      <video ref={videoRef} style={{ position: "relative" }} width="750" height="500" controls >
        <source src={isOn === true ? "/video_2240x1572.mp4" : "/video_1280x960.mp4"} type="video/mp4" />
      </video>
      <br />
      <br />
      <br />
      <a href="#ssd">SSD</a>
    </div>
  )
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
