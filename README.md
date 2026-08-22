# Conductor

Control music with your hands. Runs in a browser tab off a webcam, with nothing to install; the video never leaves the
machine.

[Try it](https://p3zo.github.io/conductor/)

## Controls

The camera frame is the instrument. A cursor follows your hand; where it sits sets the tempo and the volume, and the
pose of the hand starts and stops playback.

| Input                    | Effect                           |
| ------------------------ | -------------------------------- |
| Horizontal hand position | Playback rate, from 0.5x to 1.5x |
| Vertical hand position   | Volume, loudest at the top       |
| Pinch                    | Play. The cursor turns green     |
| Fist                     | Pause. The cursor turns red      |

Hand tracking is [Yoha](https://github.com/handtracking-io/yoha), running its models on TensorFlow.js in WebAssembly.
Every frame it returns the hand's landmarks and a probability for each pose it recognises. Playback is
[Howler](https://howlerjs.com/), which can change rate and volume on a playing track without restarting it.

Adapted from the Yoha draw demo.

## Implementation notes

**Smoothing.** Tracking a hand at video rate is noisy, so the cursor follows an exponential moving average of the
tracked position (`ExponentialCoordinateAverage(0.85)`), weighted towards where the hand already was. Reading the raw
position each frame makes the tempo shudder even while a hand is held still.

**Quantization.** Rate and volume are rounded to a tenth, and applied only when the rounded value changes. Writing a
slightly different playback rate on every frame is audible as a warble.

**Border padding.** The tracked area is cropped in from the edges of the video by `BORDER_PADDING_FACTOR`, so the
extremes of each control sit inside the frame rather than at the edge of what the camera can see, where tracking drops
out.

## Requirements

Camera permission, and enough light for the tracker to find your hand.

## Development

Run the application locally with `yarn && yarn start`.

Push changes to the `main` branch, then run `deploy.sh` to build the application on the `pages` branch where gh-pages
will serve it from.
