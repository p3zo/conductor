import * as tflite from '@tensorflow/tfjs-tflite';
import { DownloadMultipleModelBlobs, DownloadMultipleYohaModelBlobs, DownloadBlobs, } from './base';
// XXX refactor this?
import { CreateTensorFromModelInput } from './tfjs';
/**
 * @public
 * Downloads the Yoha tflite models.
 * @param boxUrl - Url to model.json file of box model.
 * @param lanUrl - Url to model.json file of landmark model.
 * @param progressCb - A callback that is called with the cumulative download progress for all
 *                     models.
 */
export async function DownloadMultipleYohaTfliteModelBlobs(boxUrl, lanUrl, progressCb) {
    return {
        ...await DownloadMultipleYohaModelBlobs(boxUrl, lanUrl, progressCb, DownloadMultipleTfliteModelBlobs),
        modelType: 'tflite'
    };
}
/**
 * @public
 * Downloads a list of tflite models.
 * @param urls - A list of URLs. Each URL must point to a model.json file.
 * @param progressCb - A callback that is called with the cumulative download progress for all
 *                     models.
 */
export async function DownloadMultipleTfliteModelBlobs(urls, progressCb) {
    return DownloadMultipleModelBlobs(urls, progressCb, DownloadTfliteModelBlobs);
}
/**
 * @public
 * Downloads a tflite models.
 * @param url - Url pointing to tflite model file.
 * @param progressCb - A callback that is called with the cumulative download progress for the
 *                     model.
 */
export async function DownloadTfliteModelBlobs(url, progressCb) {
    return DownloadBlobs([url], progressCb);
}
/**
 * Creates a tflite model from tflite model files.
 * @param modelBlobs - The model files from which to create a tflite model.
 */
export async function CreateTfliteModelFromModelBlobs(modelBlobs) {
    if (modelBlobs.blobs.size !== 1) {
        throw 'Expected tflite model to consist out of exactly one blob but got ' +
            modelBlobs.blobs.size;
    }
    const blobBuffer = await modelBlobs.blobs.values().next().value.arrayBuffer();
    const tfliteModel = await tflite.loadTFLiteModel(blobBuffer);
    return {
        model: tfliteModel,
    };
}
export function CreateModelCbFromTfliteModel(model, execAsync) {
    return async (modelInput) => {
        const t = CreateTensorFromModelInput(modelInput);
        // It seems that tfjs/tflite does not support any means to access the signature
        // included in the model. Thus we have to rely on the positioning of output tensors.
        // Note that the names in the tf.NamedTensorMap are tensor names which is orthogonal to
        // output names from the signature and tf does not provide canonical, robust way to set these 
        // tensor names.
        const namedOutputTensors = (model.model.predict(t));
        const res = [];
        for (const output of model.model.outputs) {
            res.push(namedOutputTensors[output.name]);
        }
        const coords = (execAsync ? (await res[0].array()) : res[0].arraySync())[0];
        const classes = (execAsync ? (await res[1].array()) : res[1].arraySync())[0];
        t.dispose();
        for (const resultTensor of res) {
            resultTensor.dispose();
        }
        return {
            // need to convert coordinates from [-1,1] into range [0,1]
            coordinates: coords.map(c => [(c[0] + 1) / 2, (c[1] + 1) / 2]),
            // classes is list of bernulli distributions. We just keep one value of it.
            classes: classes.map(v => v[1])
        };
    };
}
/**
 * Given a tflite model where the first input tensor is of shape [B,H,W,C].
 * returns [H,W]. Returns undefined if such tensor was not found.
 */
export function GetInputDimensionsFromTfliteModel(model) {
    if (!model.model.inputs.length) {
        return;
    }
    const dims = model.model.inputs[0].shape;
    if (dims.length !== 4) {
        return;
    }
    return [dims[1], dims[2]];
}
//# sourceMappingURL=tflite.js.map