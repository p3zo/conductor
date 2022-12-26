import * as tflite from '@tensorflow/tfjs-tflite';
import { IModelCb, IModelDownloadProgressCb, IBlobs } from './base';
/**
 * @public
 * A Tflite model.
 */
export interface ITfliteModel {
    /**
     * The native tflite model.
     */
    model: tflite.TFLiteModel;
}
/**
 * @public
 * The two models required for running the Yoha engine.
 */
export interface IYohaTfliteModelBlobs {
    /**
     * This field adds some type safety to protect against mismatching blobs/backends.
     */
    modelType: 'tflite';
    /**
     * The file blobs of the box model for detecting initial hand position within stream.
     */
    box: IBlobs;
    /**
     * The file blobs of the landmark model for detecting landmark locations and detecting hand poses.
     */
    lan: IBlobs;
}
/**
 * @public
 * Downloads the Yoha tflite models.
 * @param boxUrl - Url to model.json file of box model.
 * @param lanUrl - Url to model.json file of landmark model.
 * @param progressCb - A callback that is called with the cumulative download progress for all
 *                     models.
 */
export declare function DownloadMultipleYohaTfliteModelBlobs(boxUrl: string, lanUrl: string, progressCb: IModelDownloadProgressCb): Promise<IYohaTfliteModelBlobs>;
/**
 * @public
 * Downloads a list of tflite models.
 * @param urls - A list of URLs. Each URL must point to a model.json file.
 * @param progressCb - A callback that is called with the cumulative download progress for all
 *                     models.
 */
export declare function DownloadMultipleTfliteModelBlobs(urls: string[], progressCb: IModelDownloadProgressCb): Promise<IBlobs[]>;
/**
 * @public
 * Downloads a tflite models.
 * @param url - Url pointing to tflite model file.
 * @param progressCb - A callback that is called with the cumulative download progress for the
 *                     model.
 */
export declare function DownloadTfliteModelBlobs(url: string, progressCb: IModelDownloadProgressCb): Promise<IBlobs>;
/**
 * Creates a tflite model from tflite model files.
 * @param modelBlobs - The model files from which to create a tflite model.
 */
export declare function CreateTfliteModelFromModelBlobs(modelBlobs: IBlobs): Promise<ITfliteModel>;
export declare function CreateModelCbFromTfliteModel(model: ITfliteModel, execAsync: boolean): IModelCb;
/**
 * Given a tflite model where the first input tensor is of shape [B,H,W,C].
 * returns [H,W]. Returns undefined if such tensor was not found.
 */
export declare function GetInputDimensionsFromTfliteModel(model: ITfliteModel): number[];
