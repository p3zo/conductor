/**
 * This file contains everything backend-unrelated that is concerned with
 * - obtaining a model
 * - executing a model
 */
import { ITrackSource } from '../track_source';
export interface IModelResult {
    /**
     * [x, y] coordinates in the range [0, 1].
     */
    coordinates: number[][];
    /**
     * Probablities (range [0, 1]).
     */
    classes: number[];
}
export declare const enum ModelInputType {
    TRACK_SOURCE = "TRACK_SOURCE",
    TYPED_ARRAY = "TYPED_ARRAY"
}
/**
 * This type of input processes an HTML element. See {@link ITrackSource} for details.
 */
export interface ITrackSourceInput {
    type: ModelInputType.TRACK_SOURCE;
    data: ITrackSource;
}
export interface ITypedArrayInput {
    type: ModelInputType.TYPED_ARRAY;
    data: Uint8Array;
    shape: number[];
}
export declare type IModelInput = ITrackSourceInput | ITypedArrayInput;
export interface IModelCb {
    (input: IModelInput): Promise<IModelResult>;
}
/**
 * @public
 * Callback that periodically informs about download progress.
 * @remarks
 * <b>received</b>: Number in range [0,total]<br>
 * <b>total</b>: Number in range [0,INFINITY) that stays the same across invocations of the
 * function.
 */
export declare type IDownloadProgressCb = (received: number, total: number) => void;
export interface IDownloadProgressCbCreationFn {
    (): IDownloadProgressCb;
}
/**
 * This function can be used to track the overall download progress of multiple downloads.
 *
 * Functions that report download progress do so via a {@link IDownloadProgressCb}
 * argument. This function here also takes such a parameter but uses it to report the overall
 * download progress. Further it returns a callback that, when called, returns a
 * {@link IDownloadProgressCb} that should be used for one of the downloads.
 * @param combinedProgressCb Callback to report combined download progress.
 */
export declare function CreateProgressCbCreationCbForMultipleDownloads(combinedProgressCb: IDownloadProgressCb): IDownloadProgressCbCreationFn;
/**
 * Reads a stream and reports progress of reading process.
 * @param reader Reader for the stream to be processed.
 * @param cb Callback that reports the progress of reading the stream.
 *           `progress` is a number that represents the size of the just processed chunk.
 */
export declare function ReadStreamWithProgress(reader: ReadableStreamDefaultReader<Uint8Array>, cb: (progress: number) => void): Promise<Uint8Array>;
/**
 * Callback that downloads a model and reports download progress.
 */
interface IDownloadModelCb {
    (url: string, progressCb: IDownloadProgressCb): Promise<IBlobs>;
}
/**
 * Callback that downloads multiple models and reports total download progress.
 */
interface IDownloadModelsCb {
    (urls: string[], progressCb: IDownloadProgressCb): Promise<IBlobs[]>;
}
/**
 * Blobs associated with Yoha models.
 */
interface IYohaModelBlobs {
    box: IBlobs;
    lan: IBlobs;
}
/**
 * @public
 * Collection of blobs.
 */
export interface IBlobs {
    /**
     * Key (e.g. Url) to corresponding byte blob.
     */
    blobs: Map<string, Blob>;
}
/**
 * @public
 * Downloads Yoha models.
 * @param boxUrl - Url to model.json of box model. Must be understood by {@link downloadModelsCb}.
 * @param lanUrl - Url to model.json of landmark model. Must be understood
 *                 by {@link downloadModelsCb}.
 * @param progressCb - A callback that is called with the cumulative download progress for all
 *                     models.
 * @param downloadModelsCb - A callback that is called to download multiple models simultaneously.
 */
export declare function DownloadMultipleYohaModelBlobs(boxUrl: string, lanUrl: string, progressCb: IDownloadProgressCb, downloadModelsCb: IDownloadModelsCb): Promise<IYohaModelBlobs>;
/**
 * Downloads a list of models.
 * @param urls - A list of URLs, one for each model.
 * @param progressCb - A callback that is called with the cumulative download progress for all
 *                     models.
 * @param downloadModelCb - A callback that is called to download a model.
 */
export declare function DownloadMultipleModelBlobs(urls: string[], progressCb: IDownloadProgressCb, downloadModelCb: IDownloadModelCb): Promise<IBlobs[]>;
/**
 * Downloads a list of files, reports download progress and returns the files as map from
 * url to blobs.
 * @param urls Urls of the files to download.
 * @param progressCb Callback that informs about download progress.
 */
export declare function DownloadBlobs(urls: string[], progressCb: IDownloadProgressCb): Promise<IBlobs>;
/**
 * Creates a `fetch` compatible function that uses cached request results to process
 * any request.
 * @param cache The request cache. Has to contain an entry for every url that is going to be
 *              requested.
 */
export declare function CreateCacheReadingFetchFunc(cache: IBlobs): (requestInfo: RequestInfo) => Promise<Response>;
export {};
