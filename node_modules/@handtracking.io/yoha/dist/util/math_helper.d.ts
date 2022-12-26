/**
 * Takes a list of relative coordinates (in range [0-1]) and converts them to
 * absolute coordinates (in range [0 - (width - 1)] and [0 - (height - 1)]
 * respectively).
 */
export declare function MakeCoordsAbsolute(coords: number[][], widthPx: number, heightPx: number): number[][];
/**
 * Takes a list of absolute coordinates (in range [0 - (width - 1)] and
 * [0 - (height - 1)] respectively) and converts them to relative coordinates
 * (in range [0-1]).
 */
export declare function MakeCoordsRelative(coords: number[][], widthPx: number, heightPx: number): number[][];
/**
 * Computes l2 norm of given vector.
 */
export declare function ComputeL2Norm(a: number[]): number;
/**
 * Returns |v1 - v2|_2
 */
export declare function ComputeDistanceBetweenVectors(v1: number[], v2: number[]): number;
/**
 * Computes a - b elementwise.
 */
export declare function SubtractVectors(a: number[], b: number[]): number[];
/**
 * Calculates a * b element wise.
 */
export declare function MultiplyVectors(a: number[], b: number[]): number[];
/**
 * Returns a scaled to unit vector.
 */
export declare function NormalizeVector(a: number[]): number[];
/**
 * Calculates a / b element wise.
 */
export declare function DivideVectors(a: number[], b: number[]): number[];
/**
 * Calculates a + b element wise.
 */
export declare function AddVectors(a: number[], b: number[]): number[];
export declare class AspectRatioAwareRotation {
    private rotationCenter_;
    private rotationInRadians_;
    private aspectRatio_;
    private aspectRatioAwareRotationCenter_;
    private rotMat_;
    private reverseRotMat_;
    constructor(rotationCenter: number[], rotationInRadians: number, aspectRatio: number[]);
    private Initialize_;
    Apply(coords: number[][]): number[][];
    ApplyReverse(coords: number[][]): number[][];
    ApplyInternal_(coords: number[][], reverse: boolean): number[][];
}
/**
 * Implements
 * https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gafbbc470ce83812914a70abfb604f4326
 */
export declare function ComputeRotationMatrix2D(center: number[], angleInDegrees: number, scale: number): number[][];
export declare function DegreesToRadians(degrees: number): number;
export declare function Apply2DRotMatTo2DVec(m: number[][], v: number[]): number[];
export interface IExtremumCoords {
    minX: number[];
    maxX: number[];
    minY: number[];
    maxY: number[];
}
export declare function ComputeExtremumCoords(coords: number[][]): IExtremumCoords;
export declare function FlipCoordsHorizontally(coords: number[][]): number[][];
export declare function CoordsOutsideBox(box: number[][], coords: number[][]): number[][];
export declare function CoordsInsideBox(box: number[][], coords: number[][]): number[][];
