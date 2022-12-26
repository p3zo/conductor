/**
 * @public
 * Requests a 'post' animation frame.
 * @returns Promise that resolves right after frame has been rendered.
 */
export declare function RequestPostAnimationFrame(): Promise<void>;
/**
 * @public
 * Requests an animation frame.
 * @returns Promise that resolves right before next frame is to be rendered.
 */
export declare function RequestAnimationFrame(): Promise<void>;
