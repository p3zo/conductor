import { GraphModel } from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import { IModelCb, IDownloadProgressCb, IModelInput, IBlobs } from './base';
/**
 * @public
 * The computational backend type to use as the tfjs backend.
 */
export declare const enum TfjsBackendType {
    /**
     * WebGPU backend.
     * Currently not supported.
     */
    WEBGPU = "WEBGPU",
    /**
     * Webgl backend.
     */
    WEBGL = "WEBGL",
    /**
     * WASM backend.
     * Currently not supported.
     */
    WASM = "WASM",
    /**
     * CPU backend.
     * Currently not supported.
     */
    CPU = "CPU",
    /**
     * Experimental.
     * No backend is set explicitly by Yoha. You have to set up tfjs yourself with the backend of
     * your choice before instantiating the Yoha models.
     */
    MANUAL = "MANUAL"
}
/**
 * @public
 * A tfjs model.
 */
export interface ITfjsModel {
    /**
     * The tfjs model.
     */
    model: GraphModel;
    /**
     * The tfjs backend type with which this model was created.
     */
    backendType: TfjsBackendType;
}
/**
 * @public
 * The two models required for running the Yoha engine.
 */
export interface IYohaTfjsModelBlobs {
    /**
     * This field adds some type safety to protect against mismatching blobs/backends.
     */
    modelType: 'tfjs';
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
 * Downloads the Yoha tfjs models.
 * @param boxUrl - Url to model.json file of box model.
 * @param lanUrl - Url to model.json file of landmark model.
 * @param progressCb - A callback that is called with the cumulative download progress for all
 *                     models.
 */
export declare function DownloadMultipleYohaTfjsModelBlobs(boxUrl: string, lanUrl: string, progressCb: IDownloadProgressCb): Promise<IYohaTfjsModelBlobs>;
/**
 * @public
 * Downloads a list of tfjs models.
 * @param urls - A list of URLs. Each URL must point to a model.json file.
 * @param progressCb - A callback that is called with the cumulative download progress for all
 *                     models.
 */
export declare function DownloadTfjsModelBlobs(urls: string[], progressCb: IDownloadProgressCb): Promise<IBlobs[]>;
/**
 * @public
 * Downloads a tfjs model and reports download progress via a callback.
 * @param url - The URL to the model.json file of the tfjs model.
 * @param progressCb - Callback that informs about download progress.
 */
export declare function DownloadTfjsModel(url: string, progressCb: IDownloadProgressCb): Promise<IBlobs>;
export interface ITfjsManualBackendConfig {
    backendType: TfjsBackendType.MANUAL;
}
export interface IInternalTfjsManualBackendConfig {
    backendType: TfjsBackendType.MANUAL;
}
export interface ITfjsWebglBackendConfig {
    backendType: TfjsBackendType.WEBGL;
}
export interface IInternalTfjsWebglBackendConfig {
    backendType: TfjsBackendType.WEBGL;
}
/**
 * @public
 * Configuration that is specific to the tfjs wasm backend.
 *
 */
export interface ITfjsWasmBackendConfig {
    /**
     * See https://github.com/tensorflow/tfjs/tree/master/tfjs-backend-wasm#using-bundlers
     */
    wasmPaths: string;
}
export interface IInternalTfjsWasmBackendConfig {
    backendType: TfjsBackendType.WASM;
    wasmPaths: string;
}
/**
 * We create external and interal configs just so that users don't have to deal with
 * setting the 'backendType' fields...
 */
export declare type ITfjsBackendConfig = ITfjsWebglBackendConfig | ITfjsWasmBackendConfig | ITfjsManualBackendConfig;
export declare type IInternalTfjsBackendConfig = IInternalTfjsWebglBackendConfig | IInternalTfjsWasmBackendConfig | IInternalTfjsManualBackendConfig;
export declare const DEFAULT_TFJS_MANUAL_BACKEND_CONFIG: {
    type: TfjsBackendType;
};
export declare const DEFAULT_TFJS_WEBGL_BACKEND_CONFIG: {
    type: TfjsBackendType;
};
export declare const DEFAULT_TFJS_WASM_BACKEND_CONFIG: {
    type: TfjsBackendType;
    wasmPath: string;
};
/**
 * Creates a tfjs graph model from tfjs model files.
 * @param modelBlobs - The model files from which to create a tfjs model.
 * @param backendType - What computational backend to use for creation of the model.
 */
export declare function CreateTfjsModelFromModelBlobs(modelBlobs: IBlobs, config: IInternalTfjsBackendConfig): Promise<ITfjsModel>;
/**
 * Sets the tfjs webgl backend.
 */
export declare function SetTfjsBackendToWebgl(): Promise<void>;
/**
 * Sets the tfjs wasm backend.
 */
export declare function SetTfjsBackendToWasm(wasmPaths: string): Promise<void>;
/**
 * Creates a model callback from a tfjs graph model and information about how the model
 * is to be invoked.
 * @param model - The graph model to create a callback for.
 * @param execAsync - Whether to execute the model asynchronously.
 */
export declare function CreateModelCbFromTfjsModel(model: ITfjsModel, execAsync: boolean): IModelCb;
export declare function CreateTensorFromModelInput(mi: IModelInput): tf.Tensor<tf.Rank>;
/**
 * Given a Tfjs model where the first input tensor is of shape [B,H,W,C].
 * returns [H,W]. Returns undefined if such tensor was not found.
 */
export declare function GetInputDimensionsFromTfjsModel(model: ITfjsModel): number[];
