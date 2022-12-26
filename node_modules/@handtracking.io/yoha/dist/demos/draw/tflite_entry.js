/**
 * All imports used here are also available via the npm package.
 * https://www.npmjs.com/package/%40handtracking.io/yoha
 */
import { DownloadMultipleYohaTfliteModelBlobs, StartTfliteEngine, } from '../../entry';
import { CreateDrawDemo } from './draw';
async function Run() {
    CreateDrawDemo(async (src, config, progressCb, resultCb) => {
        const modelBlobs = await DownloadMultipleYohaTfliteModelBlobs('box/model.tflite', 'lan/model.tflite', progressCb);
        StartTfliteEngine(config, src, modelBlobs, resultCb);
    });
}
Run();
//# sourceMappingURL=tflite_entry.js.map