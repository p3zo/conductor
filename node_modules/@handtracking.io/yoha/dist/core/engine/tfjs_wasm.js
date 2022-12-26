import { StartTfjsEngine } from './tfjs_base';
/**
 * @public
 * Starts an analysis loop on a track source (e.g. a `<video>` element) using the tfjs wasm
 * backend.
 *
 * @param engineConfig - Engine configuration.
 * @param backendConfig - Backend configuration.
 * @param trackSource - The element to be analyzed.
 * @param resCb - Callback that is called with hand tracking results. The callback may be called
 *                with high frequency.
 * @param yohaModels - File blobs of the AI models required for the engine to run.
 *
 * @returns Promise that resolves with a callback that can be used to stop the analysis.
 */
export async function StartTfjsWasmEngine(engineConfig, backendConfig, trackSource, yohaModels, resCb) {
    return StartTfjsEngine(engineConfig, { ...backendConfig, backendType: "WASM" /* WASM */ }, yohaModels, trackSource, resCb);
}
//# sourceMappingURL=tfjs_wasm.js.map