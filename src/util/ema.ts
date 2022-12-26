export class ExponentialMovingAverage {
    private alpha_: number;
    private curValue_: number;

    constructor(alpha: number) {
        this.alpha_ = alpha;
    }

    Add(value: number): number {
        if (this.curValue_ === undefined || this.curValue_ === null) {
            this.curValue_ = value;
        } else {
            this.curValue_ = this.curValue_ * (1 - this.alpha_) + value * this.alpha_;
        }
        return this.curValue_;
    }

    Get(): number {
        return this.curValue_;
    }
}

export class ExponentialCoordinateAverage {
    private xAvg_: ExponentialMovingAverage;
    private yAvg_: ExponentialMovingAverage;

    constructor(alpha: number) {
        this.xAvg_ = new ExponentialMovingAverage(alpha);
        this.yAvg_ = new ExponentialMovingAverage(alpha);
    }

    Add(coord: number[]) {
        return [this.xAvg_.Add(coord[0]), this.yAvg_.Add(coord[1])];
    }
}

export function ComputeCursorPositionFromCoordinates(coords: number[][]) : number[] {
  return [(coords[3][0] + coords[7][0]) / 2, (coords[3][1] + coords[7][1]) / 2];
}

