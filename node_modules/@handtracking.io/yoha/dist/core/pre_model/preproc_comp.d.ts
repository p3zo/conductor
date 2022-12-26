/**
 * Holds information on how to transform a frame into another frame by applying
 * horizontal flip, rotation, crop.
 */
export interface IPreprocInfo {
    rotationCenter: number[];
    rotationInRadians: number;
    topLeft: number[];
    bottomRight: number[];
    flip: boolean;
}
/**
 * Given coordinates from the box model, compute image transformation information that can be used
 * to transform the frame for analysis with the landmark model.
 * @param coords - The result coordinates of the box model.
 * @param ar - [width, height] of the original frame.
 * @param slack - Specifies the amount of slack to add around the hand when cropping the next frame
 * to account for movement of the hand in unknown direction.
 * slack of 1 means 100% of the box' width and height are added as slack around the hand.
 * Changing this value requires changes in the underlying models.
 */
export declare function ComputePreprocInfoFromBoxCoords(coords: number[][], ar: number[], slack: number): IPreprocInfo;
/**
 * Similar to {@link CommputePreprocInfoFromBoxCoords}.
 */
export declare function ComputePreprocInfoFromLanCoords(coords: number[][], ar: number[], slack: number): IPreprocInfo;
export declare class SixPointRotationCalculation {
    private palmBottom_;
    private palmTop_;
    private aspectRatio_;
    private palmLine_;
    private normalizedPalmLine_;
    private mainJointLine_;
    private rotationInRadians_;
    constructor(palmBottom: number[], palmTop: number[], aspectRatio: number[]);
    GetRotationInRadians(): number;
    private Initialize_;
    private AdjustForAspectRatio_;
    private ComputePalmLine_;
    private ComputeMainJointLine_;
    private ComputeRotation_;
}
