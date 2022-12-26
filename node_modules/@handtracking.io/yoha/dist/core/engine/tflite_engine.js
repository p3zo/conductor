import { StartEngine, } from './base';
import { CreateTfliteModelFromModelBlobs, CreateModelCbFromTfliteModel, GetInputDimensionsFromTfliteModel } from '../model/tflite';
import { CreateHtmlCanvasBasedPreprocCb } from '../pre_model/canvas_preproc';
/**
 * @public
 * Starts an analysis loop on a track source (e.g. a `<video>` element) using the tflite
 * backend.
 *
 * @param config - Engine configuration.
 * @param trackSource - The element to be analyzed.
 * @param resCb - Callback that is called with hand tracking results. The callback may be called
 *                with high frequency.
 * @param yohaModels - File blobs of the AI models required for the engine to run.
 *
 * @returns Promise that resolves with a callback that can be used to stop the analysis.
 */
export async function StartTfliteEngine(config, trackSource, yohaModels, resCb) {
    const [boxModel, lanModel] = await Promise.all([
        CreateTfliteModelFromModelBlobs(yohaModels.box),
        CreateTfliteModelFromModelBlobs(yohaModels.lan),
    ]);
    const boxDims = GetInputDimensionsFromTfliteModel(boxModel);
    const lanDims = GetInputDimensionsFromTfliteModel(lanModel);
    if (boxDims[0] !== lanDims[0] || boxDims[1] !== lanDims[1]) {
        throw 'Engine does not support different dimensions for box and landmark model right now.';
    }
    const preprocCb = CreateHtmlCanvasBasedPreprocCb(trackSource.width, trackSource.height, boxDims[0], boxDims[1]);
    const boxCb = CreateModelCbFromTfliteModel(boxModel, true);
    const lanCb = CreateModelCbFromTfliteModel(lanModel, true);
    return StartEngine(config, trackSource, preprocCb, boxCb, lanCb, resCb);
}
//# sourceMappingURL=tflite_engine.js.map