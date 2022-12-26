export function IsWebglOneOrTwoAvailable() {
    return IsWebglOneAvailable() || IsWebglTwoAvailable();
}
export function IsWebglTwoAvailable() {
    return !!document.createElement('canvas').getContext('webgl2');
}
export function IsWebglOneAvailable() {
    return !!document.createElement('canvas').getContext('webgl');
}
//# sourceMappingURL=webgl_helper.js.map