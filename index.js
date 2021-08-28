/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as handpose from "@tensorflow-models/handpose";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

let model,
  videoWidth,
  videoHeight,
  ctx,
  canvas,
  audio,
  fingerLookupIndices = {
    thumb: [0, 1, 2, 3, 4],
    indexFinger: [0, 5, 6, 7, 8],
    middleFinger: [0, 9, 10, 11, 12],
    ringFinger: [0, 13, 14, 15, 16],
    pinky: [0, 17, 18, 19, 20]
  }; // for rendering each finger as a polyline

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 500;
const THUMB_TIP = 4;
const INDEX_FINGER_TIP = 8;
const mobile = isMobile();

const state = {
  backend: "webgl"
};

/*const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);*/

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}

function calculateDistance(keypoints) {
  const x1 = keypoints[THUMB_TIP][0];
  const y1 = keypoints[THUMB_TIP][1];
  const x2 = keypoints[INDEX_FINGER_TIP][0];
  const y2 = keypoints[INDEX_FINGER_TIP][1];
  const dist = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
  //console.log(dist);

  if (dist < 60 && audio.paused) {
    audio.currentTime = 0;
    audio.play();
  }
  if (dist > 60) audio.pause();
}

function drawKeypoints(keypoints) {
  for (let i = 0; i < keypoints.length; i++) {
    const x = keypoints[i][0];
    const y = keypoints[i][1];
    ctx.beginPath();
    //if (i === THUMB_TIP || i === INDEX_FINGER_TIP)
    //ctx.arc(x, y, 10, 0, 2 * Math.PI);
    //else
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fill();
  }

  const fingers = Object.keys(fingerLookupIndices);
  for (let i = 0; i < fingers.length; i++) {
    const finger = fingers[i];
    const points = fingerLookupIndices[finger].map((idx) => keypoints[idx]);
    drawPath(points, false);
  }
}

function drawPath(points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      "Browser API navigator.mediaDevices.getUserMedia not available"
    );
  }

  const video = document.getElementById("video");
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      // Only setting the video to a specified size in order to accommodate a
      // point cloud, so on mobile devices accept the default size.
      width: mobile ? undefined : VIDEO_WIDTH,
      height: mobile ? undefined : VIDEO_HEIGHT
    }
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

const landmarksRealTime = async (video) => {
  async function frameLandmarks() {
    //stats.begin();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const predictions = await model.estimateHands(video);
    if (predictions.length > 0) {
      const keypoints = predictions[0].landmarks;
      //console.log(result);
      drawKeypoints(keypoints);
      calculateDistance(keypoints);
    }
    //stats.end();
    requestAnimationFrame(frameLandmarks);
  }
  frameLandmarks();
};

navigator.getUserMedia =
  navigator.getUserMedia ||
  navigator.webkitGetUserMedia ||
  navigator.mozGetUserMedia;

if ("xr" in navigator) {
  console.log("WebXR can be used");
} else {
  console.log("WebXR isn't available");
}

async function main() {
  await tf.setBackend(state.backend);
  model = await handpose.load();
  let video;

  try {
    video = await setupCamera();
    let loaded = document.getElementById("loaded");
    let loading = document.getElementById("loading");
    loaded.style.display = "block";
    loading.style.display = "none";
  } catch (e) {
    let info = document.getElementById("info");
    info.textContent = e.message;
    info.style.display = "block";
    throw e;
  }

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;

  canvas = document.getElementById("output");
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  audio = document.getElementById("audio");

  ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, videoWidth, videoHeight);
  ctx.strokeStyle = "red";
  ctx.fillStyle = "red";

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  landmarksRealTime(video);
}

main();
