export declare class ExponentialMovingAverage {
    private alpha_;
    private curValue_;
    constructor(alpha: number);
    Add(value: number): number;
    Get(): number;
}
