import { ObjValues } from './enum_helper';
/**
 * @public
 * Result of trying to obtain a MediaStream.
 */
export interface IMediaStreamResult {
    /**
     * The created mediaStream. Set if no error occurred.
     */
    stream?: MediaStream;
    /**
     * Type of error that occurred. Set if an error occurred.
     */
    error?: ObjValues<typeof MediaStreamErrorEnum>;
}
/**
 * @public
 * Possible error types that can occur when calling navigator.mediaDevices.getUserMedia.
 *
 * @remarks
 *
 * See {@link https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia#exceptions | here}
 * for more infos.
 */
export declare const MediaStreamErrorEnum: {
    readonly ABORT_ERROR: "AbortError";
    readonly NOT_ALLOWED_ERROR: "NotAllowedError";
    readonly NOT_FOUND_ERROR: "NotFoundError";
    readonly NOT_READABLE_ERROR: "NotReadableError";
    readonly OVERCONSTRAINTED_ERROR: "OverconstrainedError";
    readonly SECURITY_ERROR: "SecurityError";
    readonly TYPE_ERROR: "TypeError";
};
/**
 * @public
 */
export interface IResolution {
    width: number;
    height: number;
}
/**
 * @public
 *
 * Creates a `<video>` element for the given stream.
 *
 * @param stream - The stream to associate with the video element.
 */
export declare function CreateVideoElementFromStream(stream: MediaStream): HTMLVideoElement;
/**
 * @public
 */
export declare function GetStreamDimensions(stream: MediaStream): IResolution;
/**
 * @public
 */
export declare function ScaleResolutionToWidth(resolution: IResolution, width: number): IResolution;
/**
 * @public
 */
export declare function ScaleResolutionToHeight(resolution: IResolution, height: number): IResolution;
/**
 * @public
 */
export declare function ScaleResolutionDown(resolution: IResolution, upperResolutionLimit: IResolution): IResolution;
/**
 * @public
 */
export declare function ScaleResolutionUp(resolution: IResolution, lowerResolutionLimit: IResolution): IResolution;
/**
 * @public
 * Keeping the aspect ratio, scales given resolution to minimize the euclidian
 * distance to the target resolution.
 */
export declare function ScaleResolutionMinimizingEuclidianDistance(resolution: IResolution, targetResolution: IResolution): IResolution;
/**
 * @public
 */
export declare function GetStreamFrameRate(stream: MediaStream): number;
/**
 * @public
 * Creates a MediaStream with maximum fps and given the max fps the highest available resolution.
 */
export declare function CreateMaxFpsMaxResStream(): Promise<IMediaStreamResult>;
