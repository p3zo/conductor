export class ExponentialMovingAverage {
    constructor(alpha) {
        this.alpha_ = alpha;
    }
    Add(value) {
        if (this.curValue_ === undefined || this.curValue_ === null) {
            this.curValue_ = value;
        }
        else {
            this.curValue_ = this.curValue_ * (1 - this.alpha_) + value * this.alpha_;
        }
        return this.curValue_;
    }
    Get() {
        return this.curValue_;
    }
}
//# sourceMappingURL=ema.js.map