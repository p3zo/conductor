/**
 * This file contains code for the draw demo. You can find a live version at
 * https://handtracking.io/draw_demo
 *
 * Note that the complete demo consists out of this file and one of the `*_entry` files contained
 * in this directory (there is one for each backend Yoha supports).
 */
/**
 * Same imports are available via the Yoha npm package
 * https://www.npmjs.com/package/%40handtracking.io/yoha
 */
import { IDownloadProgressCb, ITrackResultCb, ITrackSource, IEngineConfig } from '../../entry';
/**
 * The purpose of this interface is to encapsulate backend specific things
 * into separate files (e.g. tfjs_webgl_entry.ts) and reuse the code in this file.
 */
interface IDownloadAndStartEngineCb {
    (src: ITrackSource, config: IEngineConfig, progressCb: IDownloadProgressCb, resultCb: ITrackResultCb): void;
}
export declare function CreateDrawDemo(startCb: IDownloadAndStartEngineCb): Promise<void>;
export {};
