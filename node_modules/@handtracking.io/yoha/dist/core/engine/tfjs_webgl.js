import { StartTfjsEngine } from './tfjs_base';
/**
 * @public
 * Starts an analysis loop on a track source (e.g. a `<video>` element) using the tfjs webgl
 * backend.
 *
 * @param engineConfig - Engine configuration.
 * @param trackSource - The element to be analyzed.
 * @param resCb - Callback that is called with hand tracking results. The callback may be called
 *                with high frequency.
 * @param yohaModels - File blobs of the AI models required for the engine to run.
 *
 * @returns Promise that resolves with a callback that can be used to stop the analysis.
 */
export async function StartTfjsWebglEngine(engineConfig, trackSource, yohaModels, resCb) {
    return StartTfjsEngine(engineConfig, { backendType: "WEBGL" /* WEBGL */ }, yohaModels, trackSource, resCb);
}
//# sourceMappingURL=tfjs_webgl.js.map