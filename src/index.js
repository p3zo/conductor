/* Adapted from the Yoha draw demo: https://github.com/handtracking-io/yoha/tree/main/src/demos/draw */

import * as Tone from 'tone';
import * as yoha from '@handtracking.io/yoha';

import {
    VideoLayer,
    PointLayer,
    LayerStack,
    LandmarkLayer,
    FpsLayer
} from './util/layers';
import {ScaleResolutionToWidth} from './util/stream_helper';
import {ExponentialCoordinateAverage, ComputeCursorPositionFromCoordinates} from './util/ema';

const BORDER_PADDING_FACTOR = 0.05;
const VIDEO_WIDTH_FACTOR = 0.66;


var player = new Tone.Player({
    "url": "concerto-for-guitar.m4a",
    "loop": true,
    "volume": 5,
    "onload": () => {
        console.log('Loaded audio')
        Tone.start();
        player.sync().start();
    }
}).toDestination();

const pauseTransport = function () {
    if (Tone.Transport.state == 'started') {
        Tone.Transport.pause();
    }
}

const playTransport = function () {
    if (Tone.Transport.state !== 'started') {
        Tone.Transport.start();
    }
}

function LogError(error) {
    document.getElementById('error').innerText = error;
}

function CreateLayerStack(video, width, height) {
    const stack = new LayerStack({
        width,
        height,
        outline: '1px solid white'
    });

    const videoLayer = new VideoLayer({
        width,
        height,
        virtuallyFlipHorizontal: true,
        crop: BORDER_PADDING_FACTOR,
    }, video);
    stack.AddLayer(videoLayer);

    const pointLayer = new PointLayer({
        width,
        height,
        color: 'blue',
        radius: 8,
        fill: true,
    });
    stack.AddLayer(pointLayer);

    const landmarkLayer = new LandmarkLayer({
        width,
        height,
        color: 'white'
    });
    stack.AddLayer(landmarkLayer);

    const fpsLayer = new FpsLayer({
        width,
        height,
        color: 'white'
    });
    stack.AddLayer(fpsLayer);

    return {stack, videoLayer, pointLayer, landmarkLayer, fpsLayer};
}

async function Run() {
    // Download hand tracking models
    const modelFiles = await yoha.DownloadMultipleYohaTfjsModelBlobs(
        'box/model.json',
        'lan/model.json',
        (rec, total) => {
            if (rec / total == 1) {
                console.log('Loaded hand tracking model')
            }
        }
    );

    // Set up the video feed
    const streamRes = await yoha.CreateMaxFpsMaxResStream();

    if (streamRes.error) {
        if (streamRes.error === yoha.MediaStreamErrorEnum.NOT_ALLOWED_ERROR) {
            LogError('You denied camera access. Refresh the page if this was a mistake ' +
                'and you\'d like to try again.');
            return;
        } else if (streamRes.error === yoha.MediaStreamErrorEnum.NOT_FOUND_ERROR) {
            LogError('No camera found. For the handtracking to work you need to connect a camera. ' +
                'Refresh the page to try again.');
            return;
        } else {
            LogError(`Something went wrong when trying to access your camera (${streamRes.error}) ` +
                'You may try again by refreshing the page.');
            return;

        }
    }

    const src = yoha.CreateVideoElementFromStream(streamRes.stream);

    var width = src.width;
    var height = src.height;

    // Scale to desired size
    const targetWidth = window.innerWidth * VIDEO_WIDTH_FACTOR;
    ({width, height} = ScaleResolutionToWidth({width, height}, targetWidth));

    // Create visualization layers
    const {stack, pointLayer, landmarkLayer, fpsLayer} =
        CreateLayerStack(src, width, height);
    document.getElementById('canvas').appendChild(stack.GetEl());

    // Using a subtle exponential moving average helps to get smoother results.
    // (Setting the parameter to 1 disables the smoothing if you'd like to try without it.)
    const pos = new ExponentialCoordinateAverage(0.85);

    // Note: this path must match the path in webpack config
    const wasmConfig = {wasmPaths: './node_modules/@tensorflow/tfjs-backend-wasm/dist/'};
    const thresholds = yoha.RecommendedHandPoseProbabilityThresholds;

    // Run the engine
    const config = {
        // Webcam video is usually flipped, so we want the coordinates to be flipped as well
        mirrorX: true,
        // Crop away part of the border to prevent the user from moving out of view when reaching towards edges
        padding: BORDER_PADDING_FACTOR,
    };

    yoha.StartTfjsWasmEngine(config, wasmConfig, src, modelFiles, res => {
        fpsLayer.RegisterCall();

        document.getElementById('transport').textContent = Tone.Transport.seconds.toFixed(2);

        if (res.isHandPresentProb > thresholds.IS_HAND_PRESENT) {
            const cursorPos = pos.Add(ComputeCursorPositionFromCoordinates(res.coordinates));

            pointLayer.DrawPoint(cursorPos[0], cursorPos[1]);
            pointLayer.Render();

            if (res.poses.pinchProb > thresholds.PINCH) {
                pointLayer.setColor('green')
                playTransport()
            } else {
                pointLayer.setColor('blue')
            }

            if (res.poses.fistProb > thresholds.FIST) {
                pauseTransport()
                pointLayer.setColor('red')
            }

            landmarkLayer.Draw(res.coordinates);
            landmarkLayer.Render();
        } else {
            pointLayer.Clear();
            pointLayer.Render();
            landmarkLayer.Clear();
            landmarkLayer.Render();
        }
    });


}

var clickEvent = ('ontouchstart' in document.documentElement) ? 'touchend' : 'click';

const el = document.getElementById("launch")
el.addEventListener(clickEvent, () => {
    console.log('Launched')
    el.parentElement.removeChild(el);

    const PLAYER = new Tone.Player({
        "url": "concerto-for-guitar.m4a",
        "loop": true,
        "volume": 5,
        "onload": () => {
            console.log('Loaded audio')
            Tone.start();
            PLAYER.sync().connect(FILTER).start();
            FILTER.toDestination()
        }
    })

    Run();
})
