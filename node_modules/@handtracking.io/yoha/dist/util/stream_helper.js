/**
 * @public
 * Possible error types that can occur when calling navigator.mediaDevices.getUserMedia.
 *
 * @remarks
 *
 * See {@link https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia#exceptions | here}
 * for more infos.
 */
export const MediaStreamErrorEnum = {
    ABORT_ERROR: 'AbortError',
    NOT_ALLOWED_ERROR: 'NotAllowedError',
    NOT_FOUND_ERROR: 'NotFoundError',
    NOT_READABLE_ERROR: 'NotReadableError',
    OVERCONSTRAINTED_ERROR: 'OverconstrainedError',
    SECURITY_ERROR: 'SecurityError',
    TYPE_ERROR: 'TypeError',
};
const MEDIA_STREAM_ERROR_NAMES = new Set(Object.values(MediaStreamErrorEnum));
async function GetStreamWithConstraints(constraints) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        return {
            stream
        };
    }
    catch (e) {
        if (MEDIA_STREAM_ERROR_NAMES.has(e.name)) {
            return {
                error: e.name
            };
        }
        throw e;
    }
}
/**
 * @public
 *
 * Creates a `<video>` element for the given stream.
 *
 * @param stream - The stream to associate with the video element.
 */
export function CreateVideoElementFromStream(stream) {
    const video = document.createElement('video');
    const { width, height } = GetStreamDimensions(stream);
    video.srcObject = stream;
    video.width = width;
    video.height = height;
    video.setAttribute('autoplay', '');
    // Hack for ios.
    // https://github.com/webrtc/samples/issues/929
    video.setAttribute('playsinline', 'true');
    return video;
}
/**
 * @public
 */
export function GetStreamDimensions(stream) {
    return {
        width: stream.getVideoTracks()[0].getSettings().width,
        height: stream.getVideoTracks()[0].getSettings().height,
    };
}
/**
 * @public
 */
export function ScaleResolutionToWidth(resolution, width) {
    const cw = resolution.width;
    const ch = resolution.height;
    const tw = width;
    return {
        width: tw,
        height: ch / (cw / tw),
    };
}
/**
 * @public
 */
export function ScaleResolutionToHeight(resolution, height) {
    const cw = resolution.width;
    const ch = resolution.height;
    const th = height;
    return {
        width: cw / (ch / th),
        height: th,
    };
}
/**
 * @public
 */
export function ScaleResolutionDown(resolution, upperResolutionLimit) {
    const res = resolution;
    const uRes = upperResolutionLimit;
    if (res.width <= uRes.width && res.height <= uRes.height) {
        return res;
    }
    if (res.width - uRes.width > res.height - uRes.height) {
        return ScaleResolutionToWidth(res, uRes.width);
    }
    else {
        return ScaleResolutionToHeight(res, uRes.height);
    }
}
/**
 * @public
 */
export function ScaleResolutionUp(resolution, lowerResolutionLimit) {
    const res = resolution;
    const lRes = lowerResolutionLimit;
    if (res.width >= lRes.width && res.height >= lRes.height) {
        return res;
    }
    if (res.width - lRes.width < res.height - lRes.height) {
        return ScaleResolutionToWidth(res, lRes.width);
    }
    else {
        return ScaleResolutionToHeight(res, lRes.height);
    }
}
/**
 * @public
 * Keeping the aspect ratio, scales given resolution to minimize the euclidian
 * distance to the target resolution.
 */
export function ScaleResolutionMinimizingEuclidianDistance(resolution, targetResolution) {
    const alpha = resolution.width;
    const beta = resolution.height;
    const gamma = targetResolution.width;
    const delta = targetResolution.height;
    const y = (beta * delta + gamma * alpha) / (Math.pow(alpha, 2) + Math.pow(beta, 2));
    return {
        width: alpha * y,
        height: beta * y
    };
}
/**
 * @public
 */
export function GetStreamFrameRate(stream) {
    return stream.getVideoTracks()[0].getSettings().frameRate;
}
/**
 * @public
 * Creates a MediaStream with maximum fps and given the max fps the highest available resolution.
 */
export async function CreateMaxFpsMaxResStream() {
    const w = 4096;
    const h = 2048;
    // const w = 640;
    // const h = 480;
    const fps = [60, 30, 24];
    for (let i = 0; i < fps.length; ++i) {
        const res = await GetStreamWithConstraints({
            audio: false,
            video: {
                facingMode: 'user',
                width: {
                    ideal: w,
                },
                height: {
                    ideal: h,
                },
                frameRate: {
                    min: fps[i],
                }
            },
        });
        if (res.error === MediaStreamErrorEnum.OVERCONSTRAINTED_ERROR) {
            continue;
        }
        else {
            return res;
        }
    }
    return await GetStreamWithConstraints({
        audio: false,
        video: {
            facingMode: 'user',
            width: {
                ideal: w,
            },
            height: {
                ideal: h,
            },
        },
    });
}
//# sourceMappingURL=stream_helper.js.map