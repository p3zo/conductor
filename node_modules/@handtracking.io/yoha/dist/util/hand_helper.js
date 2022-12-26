import { MakeCoordsAbsolute, ComputeDistanceBetweenVectors } from '../util/math_helper';
export function ComputeApproximatePalmSizePx(coords, widthPx, heightPx) {
    const absCoords = MakeCoordsAbsolute(coords, widthPx, heightPx);
    const coordPairs = [
        [20, 0],
        [20, 4],
        [20, 8],
        [20, 12],
        [20, 16],
        [4, 16],
    ];
    let maxDistance = -1;
    for (const p of coordPairs) {
        maxDistance = Math.max(maxDistance, ComputeDistanceBetweenVectors(absCoords[p[0]], absCoords[p[1]]));
    }
    return maxDistance;
}
//# sourceMappingURL=hand_helper.js.map