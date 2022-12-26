import { ITrackResult } from '../post_model/post_model';
import { IPreprocessCb } from '../pre_model/preproc';
import { IModelCb } from '../model/base';
import { ITrackSource } from '../track_source';
/**
 * @public
 *
 * Result callback.
 */
export declare type ITrackResultCb = (res: ITrackResult) => void;
/**
 * The default configuration that is used to fill in any gaps in the user supplied
 * configuration.
 */
export declare const DEFAULT_CONFIG: IEngineConfig;
/**
 * @public
 * Engine configuration.
 */
export interface IEngineConfig {
    /**
     * See {@link MirrorCoordinatesHorizontally}. Default is `true`.
     */
    mirrorX?: boolean;
    /**
     * See {@link ApplyPaddingToCoordinates}. Default is `0`.
     */
    padding?: number;
    /**
     * If the classifiers probability of a hand being present falls below this threshold
     * the engine will use the box model to look out for a hand. Default is `0.5`.
     */
    minHandPresenceProbabilityThreshold?: number;
    /**
     * For internal use.
     */
    __userFriendlyCoordinateOrder?: boolean;
    /**
     * For internal use.
     */
    __boxSlack?: number;
}
/**
 * @public
 *
 * A callback that when invoked stops the engine.
 */
export interface IStopEngineCb {
    (): void;
}
/**
 * @public
 *
 * @param config - Engine configuration.
 * @param trackSource - The element to be analyzed.
 * @param preprocCb - Calllback that transforms track source (rotation/crop/resize).
 * @param boxCb - Callback that runs bounding box model.
 * @param lanCb - Callback that runs landmark model.
 * @param resCb - Callback that is called with hand tracking results. The
 * callback may be called with high frequency.
 *
 * @returns Promise that resolves with a callback that can be used to stop the
 * analysis.
 */
export declare function StartEngine(config: IEngineConfig, trackSource: ITrackSource, preprocCb: IPreprocessCb, boxCb: IModelCb, lanCb: IModelCb, resCb: ITrackResultCb): Promise<IStopEngineCb>;
/**
 * Applies any post processing to the result coordinates.
 * @param config - Engine configuration.
 * @param coords - The coordinates to transform.
 */
export declare function ApplyPostProcessingToCoordinates(config: IEngineConfig, coords: number[][]): number[][];
