import * as THREE from 'three';
/**
 * @public
 */
export interface IThreeLayerConfig {
    width: number;
    height: number;
}
/**
 * @public
 */
export declare class ThreeLayer {
    private camera_;
    private scene_;
    private renderer_;
    private config_;
    constructor(config: IThreeLayerConfig);
    Add(o: THREE.Mesh | THREE.Line): void;
    GetDomElement(): HTMLCanvasElement;
    Render(): void;
    RemoveMesh(o: THREE.Mesh | THREE.Line): void;
}
/**
 * @public
 */
export interface IPointLayerConfig {
    width: number;
    height: number;
    radius?: number;
    color?: string;
    fill?: boolean;
    lineWidth?: number;
    segments?: number;
}
/**
 * @public
 */
export declare class PointLayer implements ILayer {
    private config_;
    private threeLayer_;
    private mesh_;
    private geo_;
    private mat_;
    private meshAdded_;
    constructor(config: IPointLayerConfig);
    DrawPoint(x: number, y: number): void;
    GetEl(): HTMLCanvasElement;
    Clear(): void;
    Render(): void;
}
/**
 * @public
 */
export interface IRectLayerConfig {
    width: number;
    height: number;
    rectWidth: number;
    rectHeight: number;
    color: string;
}
/**
 * @public
 */
export declare class RectLayer implements ILayer {
    private config_;
    private threeLayer_;
    private geo_;
    private mat_;
    private mesh_;
    private meshAdded_;
    constructor(config: IRectLayerConfig);
    DrawRect(x: number, y: number, rotationInRadians?: number): void;
    GetEl(): HTMLCanvasElement;
    Clear(): void;
    Render(): void;
}
/**
 * @public
 */
export interface ILandmarkLayerConfig {
    width: number;
    height: number;
    color?: string;
    lineWidth?: number;
}
/**
 * @public
 */
export declare class LandmarkLayer implements ILayer {
    private pathLayer_;
    constructor(config: ILandmarkLayerConfig);
    Draw(coords: number[][]): void;
    GetEl(): HTMLCanvasElement;
    Clear(): void;
    Render(): void;
}
/**
 * @public
 */
export interface IVideoLayerConfig {
    crop?: number;
    width: number;
    height: number;
    virtuallyFlipHorizontal: boolean;
    fadeInSeconds?: number;
    fadeOutSeconds?: number;
}
/**
 * @public
 */
export declare class VideoLayer implements ILayer {
    private config_;
    private video_;
    private container_;
    constructor(config: IVideoLayerConfig, el: MediaStream | HTMLVideoElement);
    GetEl(): HTMLDivElement;
    FadeIn(): void;
    FadeOut(): void;
    private CreateStyles_;
    private CreateFadeInStyle_;
    private CreateFadeOutStyle_;
    private GetFadeoutClassName_;
    private GetFadeInClassName_;
}
/**
 * @public
 */
export interface IPathLayerConfig {
    width: number;
    height: number;
    color?: string;
    lineWidth?: number;
    numSmoothPoints?: number;
}
/**
 * @public
 */
export declare class PathLayer implements ILayer {
    private config_;
    private threeLayer_;
    private meshes_;
    private lines_;
    private mat_;
    constructor(config: IPathLayerConfig);
    DrawPath(path: number[][]): void;
    Clear(): void;
    Render(): void;
    GetEl(): HTMLCanvasElement;
}
/**
 * @public
 */
export interface IDynamicPathLayerConfig {
    pathLayerConfig: IPathLayerConfig;
    maxLinePoints?: number;
}
/**
 * @public
 */
export declare class DynamicPathLayer implements ILayer {
    private pathLayer_;
    private tmpPathLayer_;
    private curPath_;
    private config_;
    private miniStack_;
    constructor(config: IDynamicPathLayerConfig);
    GetEl(): HTMLDivElement;
    Clear(): void;
    Render(): void;
    AddNode(x: number, y: number): void;
    EndPath(): void;
}
/**
 * @public
 */
export interface IFpsLayerConfig {
    width: number;
    height: number;
    color?: string;
    fontSize?: string;
    timeBetweenUpdatesMs?: number;
}
/**
 * @public
 */
export declare class FpsLayer {
    private calls_;
    private ema_;
    private el_;
    private intervalHandle_;
    constructor(config: IFpsLayerConfig);
    GetEl(): HTMLDivElement;
    RegisterCall(): void;
    Stop(): void;
    private UpdateFps_;
}
/**
 * @public
 */
export interface ILayerStackConfig {
    width: number;
    height: number;
    border?: string;
    outline?: string;
}
/**
 * @public
 */
export interface ILayer {
    GetEl(): HTMLElement;
}
/**
 * @public
 */
export declare class LayerStack {
    private config_;
    private layers_;
    private containerEl_;
    constructor(config: ILayerStackConfig);
    private CreateContainerElement_;
    AddLayer(layer: ILayer, name?: string): void;
    GetEl(): HTMLDivElement;
}
