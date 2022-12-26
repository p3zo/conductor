import { IPreprocInfo } from '../pre_model/preproc_comp';
/**
 * @public
 *
 * The detected hand poses.
 *
 */
export interface IPoseProbabilities {
    /**
     * Probability of pinch gesture. Pinch means tip of index finger and tip of thumb touch.
     */
    pinchProb: number;
    /**
     * Probability of fist gesture.
     */
    fistProb: number;
}
/**
 * @public
 * Recommended thresholds for probabilities reported in {@link IPoseProbabilities},
 * {@link ITrackResult.isLeftHandProb} and {@link ITrackResult.isHandPresentProb}.
 * @privateRemarks
 * These thresholds are an optimization. Math.round(probability) works fine in general.
 * The thresholds have been determined empirically.
 */
export declare const RecommendedHandPoseProbabilityThresholds: {
    PINCH: number;
    FIST: number;
    IS_HAND_PRESENT: number;
    IS_LEFT_HAND: number;
};
/**
 * @public
 *
 * Hand tracking results.
 */
export interface ITrackResult {
    /**
     * Array of 21 [x, y] coordinates where x and y are relative coordinates such that
     * [x,y] = [0,0] corresponds to the top left pixel of the track source and [x,y] = [1,1]
     * corresponds to the bottom right pixel of the track source.
     *
     * The array is orderd as follows:
     * <ul>
     *   <li>
     *     `[0-3]`: Thumb from base to tip.<br>
     *   </li>
     *   <li>
     *     `[4-7]`: Index finger from base to tip.<br>
     *   </li>
     *   <li>
     *     `[8-11]`: Middle finger from base to tip.<br>
     *   </li>
     *   <li>
     *     `[12-15]`: Ring finger from base to tip.<br>
     *   </li>
     *   <li>
     *     `[16-19]`: Little finger from base to tip.<br>
     *   </li>
     *   <li>
     *     `[20]`: Base of hand.<br>
     *   </li>
     * </ul>
     */
    coordinates: number[][];
    /**
     * The probability that the detected hand (if any) is a left hand. Note that this is with respect
     * to the processed video which may or may not be flipped depending on how you retrieved it.
     */
    isLeftHandProb: number;
    /**
     * The probability that the analyzed frame contains a hand.
     */
    isHandPresentProb: number;
    /**
     * The detected hand poses.
     */
    poses: IPoseProbabilities;
}
/**
 * Converts landmark coordinates (that are with respect to some bounding box) to global
 * coordinates (that are with respect to the original frame).
 *
 * @param preprocInfo - The preprocessing info of the box.
 * @param aspectRatio - [width, height] of the original frame.
 * @param coords - List of landmark coordinates.
 */
export declare function ConvertLandmarkCoordinatesToGlobalCoordinates(preprocInfo: IPreprocInfo, aspectRatio: number[], coords: number[][]): number[][];
/**
 * Converts list of landmark coordinates and classifier probabilities to
 * canonical tracking result structure.
 *
 * @param coords - List of coordinates from landmark model.
 * @param classes - List of classifier probabilities from landmark model.
 */
export declare function AssembleTrackResultFromCoordinatesAndClasses(coords: number[][], classes: number[]): ITrackResult;
/**
 * @public
 * Mirrors result coordinates along the y axis i.e. `[x, y]` becomes `[1 - x, y]`.
 * @param coords - The coordinates to mirror.
 */
export declare function MirrorCoordinatesHorizontally(coords: number[][]): number[][];
/**
 * @public
 * Starting from the borders, removes a percentage of the analyzed track source.
 *
 * @example
 * Example: A video with resolution `640x480` would yield landmark coordinates where
 * `[0, 0]` and `[1, 1]` correspond to pixel values of `[0, 0]` and `[639, 639]` respectively.
 *
 * Setting padding to `0.05` would make `[0, 0]` and `[1, 1]` correspond to pixel values of
 * `[639 * 0.05, 639 * 0.05]` and `[639 - 639 * 0.05, 639 - 639 * 0.05]` respectively.
 *
 * @param padding - The padding to apply.
 * @param coords - The coordinates on which to apply the padding.
 * @returns List of coordinates with the given padding applied.
 */
export declare function ApplyPaddingToCoordinates(padding: number, coords: number[][]): number[][];
/**
 * Reorders coordinates to have a more intuitive order (as opposed to the one coming from the
 * model).
 */
export declare function MakeCoordinateOrderUserFriendly(coords: number[][]): number[][];
