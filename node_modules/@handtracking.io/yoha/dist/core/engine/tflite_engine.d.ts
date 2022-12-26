import { IEngineConfig, ITrackResultCb, IStopEngineCb } from './base';
import { ITrackSource } from '../track_source';
import { IYohaTfliteModelBlobs } from '../model/tflite';
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
export declare function StartTfliteEngine(config: IEngineConfig, trackSource: ITrackSource, yohaModels: IYohaTfliteModelBlobs, resCb: ITrackResultCb): Promise<IStopEngineCb>;
