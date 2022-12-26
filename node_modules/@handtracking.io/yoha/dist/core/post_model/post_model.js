import { AspectRatioAwareRotation, CoordsOutsideBox, FlipCoordsHorizontally, CoordsInsideBox } from '../../util/math_helper';
const RECOMMENDED_CLASSIFIER_THRESHOLDS = {
    PINCH: 0.2,
    FIST: 0.5,
    IS_HAND_PRESENT: 0.5,
    IS_LEFT_HAND: 0.5,
};
/**
 * @public
 * Recommended thresholds for probabilities reported in {@link IPoseProbabilities},
 * {@link ITrackResult.isLeftHandProb} and {@link ITrackResult.isHandPresentProb}.
 * @privateRemarks
 * These thresholds are an optimization. Math.round(probability) works fine in general.
 * The thresholds have been determined empirically.
 */
export const RecommendedHandPoseProbabilityThresholds = RECOMMENDED_CLASSIFIER_THRESHOLDS;
const USER_FRIENDLY_COORD_ORDER = [
    17,
    16,
    18,
    19,
    1,
    0,
    2,
    3,
    5,
    4,
    6,
    7,
    9,
    8,
    10,
    11,
    13,
    12,
    14,
    15,
    20,
];
/**
 * Converts landmark coordinates (that are with respect to some bounding box) to global
 * coordinates (that are with respect to the original frame).
 *
 * @param preprocInfo - The preprocessing info of the box.
 * @param aspectRatio - [width, height] of the original frame.
 * @param coords - List of landmark coordinates.
 */
export function ConvertLandmarkCoordinatesToGlobalCoordinates(preprocInfo, aspectRatio, coords) {
    const p = preprocInfo;
    const araRot = new AspectRatioAwareRotation(p.rotationCenter, p.rotationInRadians, aspectRatio);
    coords = CoordsOutsideBox([p.topLeft, p.bottomRight], coords);
    coords = araRot.ApplyReverse(coords);
    return coords;
}
/**
 * Converts list of landmark coordinates and classifier probabilities to
 * canonical tracking result structure.
 *
 * @param coords - List of coordinates from landmark model.
 * @param classes - List of classifier probabilities from landmark model.
 */
export function AssembleTrackResultFromCoordinatesAndClasses(coords, classes) {
    return {
        coordinates: coords,
        poses: {
            pinchProb: classes[0],
            fistProb: classes[1],
        },
        isLeftHandProb: classes[3],
        // The class for hand presence is special case as it actually predicts 1
        // if there is no hand present so we need to invert the probability before applying.
        // the threshold.
        isHandPresentProb: (1 - classes[2]),
    };
}
/**
 * @public
 * Mirrors result coordinates along the y axis i.e. `[x, y]` becomes `[1 - x, y]`.
 * @param coords - The coordinates to mirror.
 */
export function MirrorCoordinatesHorizontally(coords) {
    return FlipCoordsHorizontally(coords);
}
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
export function ApplyPaddingToCoordinates(padding, coords) {
    const d = padding;
    const topLeft = [d, d];
    const bottomRight = [1 - d, 1 - d];
    coords = CoordsInsideBox([topLeft, bottomRight], coords);
    return coords;
}
/**
 * Reorders coordinates to have a more intuitive order (as opposed to the one coming from the
 * model).
 */
export function MakeCoordinateOrderUserFriendly(coords) {
    const newCoords = [];
    for (const i of USER_FRIENDLY_COORD_ORDER) {
        newCoords.push(coords[i]);
    }
    return newCoords;
}
//# sourceMappingURL=post_model.js.map