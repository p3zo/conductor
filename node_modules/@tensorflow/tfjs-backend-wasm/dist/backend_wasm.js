/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import './flags_wasm';
import { DataStorage, deprecationWarn, engine, env, KernelBackend, util } from '@tensorflow/tfjs-core';
import * as wasmFactoryThreadedSimd_import from '../wasm-out/tfjs-backend-wasm-threaded-simd.js';
// @ts-ignore
import { wasmWorkerContents } from '../wasm-out/tfjs-backend-wasm-threaded-simd.worker.js';
import * as wasmFactory_import from '../wasm-out/tfjs-backend-wasm.js';
// This workaround is required for importing in Node.js without using
// the node bundle (for testing). This would not be necessary if we
// flipped esModuleInterop to true, but we likely can't do that since
// google3 does not use it.
const wasmFactoryThreadedSimd = (wasmFactoryThreadedSimd_import.default
    || wasmFactoryThreadedSimd_import);
const wasmFactory = (wasmFactory_import.default
    || wasmFactory_import);
export class BackendWasm extends KernelBackend {
    constructor(wasm) {
        super();
        this.wasm = wasm;
        // 0 is reserved for null data ids.
        this.dataIdNextNumber = 1;
        this.wasm.tfjs.initWithThreadsCount(threadsCount);
        actualThreadsCount = this.wasm.tfjs.getThreadsCount();
        this.dataIdMap = new DataStorage(this, engine());
    }
    write(values, shape, dtype) {
        const dataId = { id: this.dataIdNextNumber++ };
        this.move(dataId, values, shape, dtype, 1);
        return dataId;
    }
    numDataIds() {
        return this.dataIdMap.numDataIds();
    }
    async time(f) {
        const start = util.now();
        f();
        const kernelMs = util.now() - start;
        return { kernelMs };
    }
    move(dataId, values, shape, dtype, refCount) {
        const id = this.dataIdNextNumber++;
        if (dtype === 'string') {
            const stringBytes = values;
            this.dataIdMap.set(dataId, { id, stringBytes, shape, dtype, memoryOffset: null, refCount });
            return;
        }
        const size = util.sizeFromShape(shape);
        const numBytes = size * util.bytesPerElement(dtype);
        const memoryOffset = this.wasm._malloc(numBytes);
        this.dataIdMap.set(dataId, { id, memoryOffset, shape, dtype, refCount });
        this.wasm.tfjs.registerTensor(id, size, memoryOffset);
        if (values != null) {
            this.wasm.HEAPU8.set(new Uint8Array(values.buffer, values.byteOffset, numBytes), memoryOffset);
        }
    }
    async read(dataId) {
        return this.readSync(dataId);
    }
    readSync(dataId, start, end) {
        const { memoryOffset, dtype, shape, stringBytes } = this.dataIdMap.get(dataId);
        if (dtype === 'string') {
            // Slice all elements.
            if ((start == null || start === 0) &&
                (end == null || end >= stringBytes.length)) {
                return stringBytes;
            }
            return stringBytes.slice(start, end);
        }
        start = start || 0;
        end = end || util.sizeFromShape(shape);
        const bytesPerElement = util.bytesPerElement(dtype);
        const bytes = this.wasm.HEAPU8.slice(memoryOffset + start * bytesPerElement, memoryOffset + end * bytesPerElement);
        return typedArrayFromBuffer(bytes.buffer, dtype);
    }
    /**
     * Dispose the memory if the dataId has 0 refCount. Return true if the memory
     * is released, false otherwise.
     * @param dataId
     * @oaram force Optional, remove the data regardless of refCount
     */
    disposeData(dataId, force = false) {
        if (this.dataIdMap.has(dataId)) {
            const data = this.dataIdMap.get(dataId);
            data.refCount--;
            if (!force && data.refCount > 0) {
                return false;
            }
            this.wasm._free(data.memoryOffset);
            this.wasm.tfjs.disposeData(data.id);
            this.dataIdMap.delete(dataId);
        }
        return true;
    }
    /** Return refCount of a `TensorData`. */
    refCount(dataId) {
        if (this.dataIdMap.has(dataId)) {
            const tensorData = this.dataIdMap.get(dataId);
            return tensorData.refCount;
        }
        return 0;
    }
    incRef(dataId) {
        const data = this.dataIdMap.get(dataId);
        if (data != null) {
            data.refCount++;
        }
    }
    floatPrecision() {
        return 32;
    }
    // Returns the memory offset of a tensor. Useful for debugging and unit
    // testing.
    getMemoryOffset(dataId) {
        return this.dataIdMap.get(dataId).memoryOffset;
    }
    dispose() {
        this.wasm.tfjs.dispose();
        if ('PThread' in this.wasm) {
            this.wasm.PThread.terminateAllThreads();
        }
        this.wasm = null;
    }
    memory() {
        return { unreliable: false };
    }
    /**
     * Make a tensor info for the output of an op. If `memoryOffset` is not
     * present, this method allocates memory on the WASM heap. If `memoryOffset`
     * is present, the memory was allocated elsewhere (in c++) and we just record
     * the pointer where that memory lives.
     */
    makeOutput(shape, dtype, memoryOffset) {
        let dataId;
        if (memoryOffset == null) {
            dataId = this.write(null /* values */, shape, dtype);
        }
        else {
            const id = this.dataIdNextNumber++;
            dataId = { id };
            this.dataIdMap.set(dataId, { id, memoryOffset, shape, dtype, refCount: 1 });
            const size = util.sizeFromShape(shape);
            this.wasm.tfjs.registerTensor(id, size, memoryOffset);
        }
        return { dataId, shape, dtype };
    }
    typedArrayFromHeap({ shape, dtype, dataId }) {
        const buffer = this.wasm.HEAPU8.buffer;
        const { memoryOffset } = this.dataIdMap.get(dataId);
        const size = util.sizeFromShape(shape);
        switch (dtype) {
            case 'float32':
                return new Float32Array(buffer, memoryOffset, size);
            case 'int32':
                return new Int32Array(buffer, memoryOffset, size);
            case 'bool':
                return new Uint8Array(buffer, memoryOffset, size);
            default:
                throw new Error(`Unknown dtype ${dtype}`);
        }
    }
}
function createInstantiateWasmFunc(path) {
    // this will be replace by rollup plugin patchWechatWebAssembly in
    // minprogram's output.
    // tslint:disable-next-line:no-any
    return (imports, callback) => {
        util.fetch(path, { credentials: 'same-origin' }).then((response) => {
            if (!response['ok']) {
                imports.env.a(`failed to load wasm binary file at '${path}'`);
            }
            response.arrayBuffer().then(binary => {
                WebAssembly.instantiate(binary, imports).then(output => {
                    callback(output.instance, output.module);
                });
            });
        });
        return {};
    };
}
/**
 * Returns the path of the WASM binary.
 * @param simdSupported whether SIMD is supported
 * @param threadsSupported whether multithreading is supported
 * @param wasmModuleFolder the directory containing the WASM binaries.
 */
function getPathToWasmBinary(simdSupported, threadsSupported, wasmModuleFolder) {
    if (wasmPath != null) {
        // If wasmPath is defined, the user has supplied a full path to
        // the vanilla .wasm binary.
        return wasmPath;
    }
    let path = 'tfjs-backend-wasm.wasm';
    if (simdSupported && threadsSupported) {
        path = 'tfjs-backend-wasm-threaded-simd.wasm';
    }
    else if (simdSupported) {
        path = 'tfjs-backend-wasm-simd.wasm';
    }
    if (wasmFileMap != null) {
        if (wasmFileMap[path] != null) {
            return wasmFileMap[path];
        }
    }
    return wasmModuleFolder + path;
}
/**
 * Initializes the wasm module and creates the js <--> wasm bridge.
 *
 * NOTE: We wrap the wasm module in a object with property 'wasm' instead of
 * returning Promise<BackendWasmModule> to avoid freezing Chrome (last tested
 * in Chrome 76).
 */
export async function init() {
    const [simdSupported, threadsSupported] = await Promise.all([
        env().getAsync('WASM_HAS_SIMD_SUPPORT'),
        env().getAsync('WASM_HAS_MULTITHREAD_SUPPORT')
    ]);
    return new Promise((resolve, reject) => {
        const factoryConfig = {};
        /**
         * This function overrides the Emscripten module locateFile utility.
         * @param path The relative path to the file that needs to be loaded.
         * @param prefix The path to the main JavaScript file's directory.
         */
        factoryConfig.locateFile = (path, prefix) => {
            if (path.endsWith('.worker.js')) {
                // Escape '\n' because Blob will turn it into a newline.
                // There should be a setting for this, but 'endings: "native"' does
                // not seem to work.
                const response = wasmWorkerContents.replace(/\n/g, '\\n');
                const blob = new Blob([response], { type: 'application/javascript' });
                return URL.createObjectURL(blob);
            }
            if (path.endsWith('.wasm')) {
                return getPathToWasmBinary(simdSupported, threadsSupported, wasmPathPrefix != null ? wasmPathPrefix : prefix);
            }
            return prefix + path;
        };
        // Use the instantiateWasm override when system fetch is not available.
        // Reference:
        // https://github.com/emscripten-core/emscripten/blob/2bca083cbbd5a4133db61fbd74d04f7feecfa907/tests/manual_wasm_instantiate.html#L170
        if (customFetch) {
            factoryConfig.instantiateWasm =
                createInstantiateWasmFunc(getPathToWasmBinary(simdSupported, threadsSupported, wasmPathPrefix != null ? wasmPathPrefix : ''));
        }
        let initialized = false;
        factoryConfig.onAbort = () => {
            if (initialized) {
                // Emscripten already called console.warn so no need to double log.
                return;
            }
            if (initAborted) {
                // Emscripten calls `onAbort` twice, resulting in double error
                // messages.
                return;
            }
            initAborted = true;
            const rejectMsg = 'Make sure the server can serve the `.wasm` file relative to the ' +
                'bundled js file. For more details see https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers';
            reject({ message: rejectMsg });
        };
        let wasm;
        // If `wasmPath` has been defined we must initialize the vanilla module.
        if (threadsSupported && simdSupported && wasmPath == null) {
            factoryConfig.mainScriptUrlOrBlob = new Blob([`var WasmBackendModuleThreadedSimd = ` +
                    wasmFactoryThreadedSimd.toString()], { type: 'text/javascript' });
            wasm = wasmFactoryThreadedSimd(factoryConfig);
        }
        else {
            // The wasmFactory works for both vanilla and SIMD binaries.
            wasm = wasmFactory(factoryConfig);
        }
        // The `wasm` promise will resolve to the WASM module created by
        // the factory, but it might have had errors during creation. Most
        // errors are caught by the onAbort callback defined above.
        // However, some errors, such as those occurring from a
        // failed fetch, result in this promise being rejected. These are
        // caught and re-rejected below.
        wasm.then((module) => {
            initialized = true;
            initAborted = false;
            const voidReturnType = null;
            // Using the tfjs namespace to avoid conflict with emscripten's API.
            module.tfjs = {
                init: module.cwrap('init', null, []),
                initWithThreadsCount: module.cwrap('init_with_threads_count', null, ['number']),
                getThreadsCount: module.cwrap('get_threads_count', 'number', []),
                registerTensor: module.cwrap('register_tensor', null, [
                    'number',
                    'number',
                    'number',
                ]),
                disposeData: module.cwrap('dispose_data', voidReturnType, ['number']),
                dispose: module.cwrap('dispose', voidReturnType, []),
            };
            resolve({ wasm: module });
        }).catch(reject);
    });
}
function typedArrayFromBuffer(buffer, dtype) {
    switch (dtype) {
        case 'float32':
            return new Float32Array(buffer);
        case 'int32':
            return new Int32Array(buffer);
        case 'bool':
            return new Uint8Array(buffer);
        default:
            throw new Error(`Unknown dtype ${dtype}`);
    }
}
const wasmBinaryNames = [
    'tfjs-backend-wasm.wasm', 'tfjs-backend-wasm-simd.wasm',
    'tfjs-backend-wasm-threaded-simd.wasm'
];
let wasmPath = null;
let wasmPathPrefix = null;
let wasmFileMap = {};
let initAborted = false;
let customFetch = false;
/**
 * @deprecated Use `setWasmPaths` instead.
 * Sets the path to the `.wasm` file which will be fetched when the wasm
 * backend is initialized. See
 * https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm/README.md#using-bundlers
 * for more details.
 * @param path wasm file path or url
 * @param usePlatformFetch optional boolean to use platform fetch to download
 *     the wasm file, default to false.
 *
 * @doc {heading: 'Environment', namespace: 'wasm'}
 */
export function setWasmPath(path, usePlatformFetch = false) {
    deprecationWarn('setWasmPath has been deprecated in favor of setWasmPaths and' +
        ' will be removed in a future release.');
    if (initAborted) {
        throw new Error('The WASM backend was already initialized. Make sure you call ' +
            '`setWasmPath()` before you call `tf.setBackend()` or `tf.ready()`');
    }
    wasmPath = path;
    customFetch = usePlatformFetch;
}
/**
 * Configures the locations of the WASM binaries.
 *
 * ```js
 * setWasmPaths({
 *  'tfjs-backend-wasm.wasm': 'renamed.wasm',
 *  'tfjs-backend-wasm-simd.wasm': 'renamed-simd.wasm',
 *  'tfjs-backend-wasm-threaded-simd.wasm': 'renamed-threaded-simd.wasm'
 * });
 * tf.setBackend('wasm');
 * ```
 *
 * @param prefixOrFileMap This can be either a string or object:
 *  - (string) The path to the directory where the WASM binaries are located.
 *     Note that this prefix will be used to load each binary (vanilla,
 *     SIMD-enabled, threading-enabled, etc.).
 *  - (object) Mapping from names of WASM binaries to custom
 *     full paths specifying the locations of those binaries. This is useful if
 *     your WASM binaries are not all located in the same directory, or if your
 *     WASM binaries have been renamed.
 * @param usePlatformFetch optional boolean to use platform fetch to download
 *     the wasm file, default to false.
 *
 * @doc {heading: 'Environment', namespace: 'wasm'}
 */
export function setWasmPaths(prefixOrFileMap, usePlatformFetch = false) {
    if (initAborted) {
        throw new Error('The WASM backend was already initialized. Make sure you call ' +
            '`setWasmPaths()` before you call `tf.setBackend()` or ' +
            '`tf.ready()`');
    }
    if (typeof prefixOrFileMap === 'string') {
        wasmPathPrefix = prefixOrFileMap;
    }
    else {
        wasmFileMap = prefixOrFileMap;
        const missingPaths = wasmBinaryNames.filter(name => wasmFileMap[name] == null);
        if (missingPaths.length > 0) {
            throw new Error(`There were no entries found for the following binaries: ` +
                `${missingPaths.join(',')}. Please either call setWasmPaths with a ` +
                `map providing a path for each binary, or with a string indicating ` +
                `the directory where all the binaries can be found.`);
        }
    }
    customFetch = usePlatformFetch;
}
/** Used in unit tests. */
export function resetWasmPath() {
    wasmPath = null;
    wasmPathPrefix = null;
    wasmFileMap = {};
    customFetch = false;
    initAborted = false;
}
let threadsCount = -1;
let actualThreadsCount = -1;
/**
 * Sets the number of threads that will be used by XNNPACK to create
 * threadpool (default to the number of logical CPU cores).
 *
 * This must be called before calling `tf.setBackend('wasm')`.
 */
export function setThreadsCount(numThreads) {
    threadsCount = numThreads;
}
/**
 * Gets the actual threads count that is used by XNNPACK.
 *
 * It is set after the backend is intialized.
 */
export function getThreadsCount() {
    if (actualThreadsCount === -1) {
        throw new Error(`WASM backend not initialized.`);
    }
    return actualThreadsCount;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmFja2VuZF93YXNtLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdhc20vc3JjL2JhY2tlbmRfd2FzbS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFDSCxPQUFPLGNBQWMsQ0FBQztBQUV0QixPQUFPLEVBQWtDLFdBQVcsRUFBWSxlQUFlLEVBQUUsTUFBTSxFQUFFLEdBQUcsRUFBRSxhQUFhLEVBQWMsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFJNUosT0FBUSxLQUFLLDhCQUE4QixNQUFNLGdEQUFnRCxDQUFDO0FBQ2xHLGFBQWE7QUFDYixPQUFPLEVBQUMsa0JBQWtCLEVBQUMsTUFBTSx1REFBdUQsQ0FBQztBQUN6RixPQUFPLEtBQUssa0JBQWtCLE1BQU0sa0NBQWtDLENBQUM7QUFFdkUscUVBQXFFO0FBQ3JFLG1FQUFtRTtBQUNuRSxxRUFBcUU7QUFDckUsMkJBQTJCO0FBQzNCLE1BQU0sdUJBQXVCLEdBQUcsQ0FBQyw4QkFBOEIsQ0FBQyxPQUFPO09BQ2xFLDhCQUE4QixDQUNVLENBQUM7QUFDOUMsTUFBTSxXQUFXLEdBQUcsQ0FBQyxrQkFBa0IsQ0FBQyxPQUFPO09BQzFDLGtCQUFrQixDQUFzQyxDQUFDO0FBYzlELE1BQU0sT0FBTyxXQUFZLFNBQVEsYUFBYTtJQUs1QyxZQUFtQixJQUFxRDtRQUN0RSxLQUFLLEVBQUUsQ0FBQztRQURTLFNBQUksR0FBSixJQUFJLENBQWlEO1FBSnhFLG1DQUFtQztRQUMzQixxQkFBZ0IsR0FBRyxDQUFDLENBQUM7UUFLM0IsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsb0JBQW9CLENBQUMsWUFBWSxDQUFDLENBQUM7UUFDbEQsa0JBQWtCLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsZUFBZSxFQUFFLENBQUM7UUFDdEQsSUFBSSxDQUFDLFNBQVMsR0FBRyxJQUFJLFdB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QUFDO0lBQ3RCLFdBQVcsR0FBRyxFQUFFLENBQUM7SUFDakIsV0FBVyxHQUFHLEtBQUssQ0FBQztJQUNwQixXQUFXLEdBQUcsS0FBSyxDQUFDO0FBQ3RCLENBQUM7QUFFRCxJQUFJLFlBQVksR0FBRyxDQUFDLENBQUMsQ0FBQztBQUN0QixJQUFJLGtCQUFrQixHQUFHLENBQUMsQ0FBQyxDQUFDO0FBRTVCOzs7OztHQUtHO0FBQ0gsTUFBTSxVQUFVLGVBQWUsQ0FBQyxVQUFrQjtJQUNoRCxZQUFZLEdBQUcsVUFBVSxDQUFDO0FBQzVCLENBQUM7QUFFRDs7OztHQUlHO0FBQ0gsTUFBTSxVQUFVLGVBQWU7SUFDN0IsSUFBSSxrQkFBa0IsS0FBSyxDQUFDLENBQUMsRUFBRTtRQUM3QixNQUFNLElBQUksS0FBSyxDQUFDLCtCQUErQixDQUFDLENBQUM7S0FDbEQ7SUFDRCxPQUFPLGtCQUFrQixDQUFDO0FBQzVCLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5pbXBvcnQgJy4vZmxhZ3Nfd2FzbSc7XG5cbmltcG9ydCB7YmFja2VuZF91dGlsLCBCYWNrZW5kVGltaW5nSW5mbywgRGF0YVN0b3JhZ2UsIERhdGFUeXBlLCBkZXByZWNhdGlvbldhcm4sIGVuZ2luZSwgZW52LCBLZXJuZWxCYWNrZW5kLCBUZW5zb3JJbmZvLCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge0JhY2tlbmRXYXNtTW9kdWxlLCBXYXNtRmFjdG9yeUNvbmZpZ30gZnJvbSAnLi4vd2FzbS1vdXQvdGZqcy1iYWNrZW5kLXdhc20nO1xuaW1wb3J0IHtCYWNrZW5kV2FzbVRocmVhZGVkU2ltZE1vZHVsZX0gZnJvbSAnLi4vd2FzbS1vdXQvdGZqcy1iYWNrZW5kLXdhc20tdGhyZWFkZWQtc2ltZCc7XG5pbXBvcnQgICogYXMgd2FzbUZhY3RvcnlUaHJlYWRlZFNpbWRfaW1wb3J0IGZyb20gJy4uL3dhc20tb3V0L3RmanMtYmFja2VuZC13YXNtLXRocmVhZGVkLXNpbWQuanMnO1xuLy8gQHRzLWlnbm9yZVxuaW1wb3J0IHt3YXNtV29ya2VyQ29udGVudHN9IGZyb20gJy4uL3dhc20tb3V0L3RmanMtYmFja2VuZC13YXNtLXRocmVhZGVkLXNpbWQud29ya2VyLmpzJztcbmltcG9ydCAqIGFzIHdhc21GYWN0b3J5X2ltcG9ydCBmcm9tICcuLi93YXNtLW91dC90ZmpzLWJhY2tlbmQtd2FzbS5qcyc7XG5cbi8vIFRoaXMgd29ya2Fyb3VuZCBpcyByZXF1aXJlZCBmb3IgaW1wb3J0aW5nIGluIE5vZGUuanMgd2l0aG91dCB1c2luZ1xuLy8gdGhlIG5vZGUgYnVuZGxlIChmb3IgdGVzdGluZykuIFRoaXMgd291bGQgbm90IGJlIG5lY2Vzc2FyeSBpZiB3ZVxuLy8gZmxpcHBlZCBlc01vZHVsZUludGVyb3AgdG8gdHJ1ZSwgYnV0IHdlIGxpa2VseSBjYW4ndCBkbyB0aGF0IHNpbmNlXG4vLyBnb29nbGUzIGRvZXMgbm90IHVzZSBpdC5cbmNvbnN0IHdhc21GYWN0b3J5VGhyZWFkZWRTaW1kID0gKHdhc21GYWN0b3J5VGhyZWFkZWRTaW1kX2ltcG9ydC5kZWZhdWx0XG4gIHx8IHdhc21GYWN0b3J5VGhyZWFkZWRTaW1kX2ltcG9ydCkgYXNcbnR5cGVvZiB3YXNtRmFjdG9yeVRocmVhZGVkU2ltZF9pbXBvcnQuZGVmYXVsdDtcbmNvbnN0IHdhc21GYWN0b3J5ID0gKHdhc21GYWN0b3J5X2ltcG9ydC5kZWZhdWx0XG4gIHx8IHdhc21GYWN0b3J5X2ltcG9ydCkgYXMgdHlwZW9mIHdhc21GYWN0b3J5X2ltcG9ydC5kZWZhdWx0O1xuXG5pbnRlcmZhY2UgVGVuc29yRGF0YSB7XG4gIGlkOiBudW1iZXI7XG4gIG1lbW9yeU9mZnNldDogbnVtYmVyO1xuICBzaGFwZTogbnVtYmVyW107XG4gIGR0eXBlOiBEYXRhVHlwZTtcbiAgcmVmQ291bnQ6IG51bWJlcjtcbiAgLyoqIE9ubHkgdXNlZCBmb3Igc3RyaW5nIHRlbnNvcnMsIHN0b3JpbmcgZW5jb2RlZCBieXRlcy4gKi9cbiAgc3RyaW5nQnl0ZXM/OiBVaW50OEFycmF5W107XG59XG5cbmV4cG9ydCB0eXBlIERhdGFJZCA9IG9iamVjdDsgIC8vIG9iamVjdCBpbnN0ZWFkIG9mIHt9IHRvIGZvcmNlIG5vbi1wcmltaXRpdmUuXG5cbmV4cG9ydCBjbGFzcyBCYWNrZW5kV2FzbSBleHRlbmRzIEtlcm5lbEJhY2tlbmQge1xuICAvLyAwIGlzIHJlc2VydmVkIGZvciBudWxsIGRhdGEgaWRzLlxuICBwcml2YXRlIGRhdGFJZE5leHROdW1iZXIgPSAxO1xuICBkYXRhSWRNYXA6IERhdGFTdG9yYWdlPFRlbnNvckRhdGE+O1xuXG4gIGNvbnN0cnVjdG9yKHB1YmxpYyB3YXNtOiBCYWNrZW5kV2FzbU1vZHVsZXxCYWNrZW5kV2FzbVRocmVhZGVkU2ltZE1vZHVsZSkge1xuICAgIHN1cGVyKCk7XG4gICAgdGhpcy53YXNtLnRmanMuaW5pdFdpdGhUaHJlYWRzQ291bnQodGhyZWFkc0NvdW50KTtcbiAgICBhY3R1YWxUaHJlYWRzQ291bnQgPSB0aGlzLndhc20udGZqcy5nZXRUaHJlYWRzQ291bnQoKTtcbiAgICB0aGlzLmRhdGFJZE1hcCA9IG5ldyBEYXRhU3RvcmFnZSh0aGlzLCBlbmdpbmUoKSk7XG4gIH1cblxuICB3cml0ZSh2YWx1ZXM6IGJhY2tlbmRfdXRpbC5CYWNrZW5kVmFsdWVzLCBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSk6XG4gICAgICBEYXRhSWQge1xuICAgIGNvbnN0IGRhdGFJZCA9IHtpZDogdGhpcy5kYXRhSWROZXh0TnVtYmVyKyt9O1xuICAgIHRoaXMubW92ZShkYXRhSWQsIHZhbHVlcywgc2hhcGUsIGR0eXBlLCAxKTtcbiAgICByZXR1cm4gZGF0YUlkO1xuICB9XG5cbiAgbnVtRGF0YUlkcygpOiBudW1iZXIge1xuICAgIHJldHVybiB0aGlzLmRhdGFJZE1hcC5udW1EYXRhSWRzKCk7XG4gIH1cblxuICBhc3luYyB0aW1lKGY6ICgpID0+IHZvaWQpOiBQcm9taXNlPEJhY2tlbmRUaW1pbmdJbmZvPiB7XG4gICAgY29uc3Qgc3RhcnQgPSB1dGlsLm5vdygpO1xuICAgIGYoKTtcbiAgICBjb25zdCBrZXJuZWxNcyA9IHV0aWwubm93KCkgLSBzdGFydDtcbiAgICByZXR1cm4ge2tlcm5lbE1zfTtcbiAgfVxuXG4gIG1vdmUoXG4gICAgICBkYXRhSWQ6IERhdGFJZCwgdmFsdWVzOiBiYWNrZW5kX3V0aWwuQmFja2VuZFZhbHVlcywgc2hhcGU6IG51bWJlcltdLFxuICAgICAgZHR5cGU6IERhdGFUeXBlLCByZWZDb3VudDogbnVtYmVyKTogdm9pZCB7XG4gICAgY29uc3QgaWQgPSB0aGlzLmRhdGFJZE5leHROdW1iZXIrKztcbiAgICBpZiAoZHR5cGUgPT09ICdzdHJpbmcnKSB7XG4gICAgICBjb25zdCBzdHJpbmdCeXRlcyA9IHZhbHVlcyBhcyBVaW50OEFycmF5W107XG4gICAgICB0aGlzLmRhdGFJZE1hcC5zZXQoXG4gICAgICAgICAgZGF0YUlkLFxuICAgICAgICAgIHtpZCwgc3RyaW5nQnl0ZXMsIHNoYXBlLCBkdHlwZSwgbWVtb3J5T2Zmc2V0OiBudWxsLCByZWZDb3VudH0pO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGNvbnN0IHNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUoc2hhcGUpO1xuICAgIGNvbnN0IG51bUJ5dGVzID0gc2l6ZSAqIHV0aWwuYnl0ZXNQZXJFbGVtZW50KGR0eXBlKTtcbiAgICBjb25zdCBtZW1vcnlPZmZzZXQgPSB0aGlzLndhc20uX21hbGxvYyhudW1CeXRlcyk7XG5cbiAgICB0aGlzLmRhdGFJZE1hcC5zZXQoZGF0YUlkLCB7aWQsIG1lbW9yeU9mZnNldCwgc2hhcGUsIGR0eXBlLCByZWZDb3VudH0pO1xuXG4gICAgdGhpcy53YXNtLnRmanMucmVnaXN0ZXJUZW5zb3IoaWQsIHNpemUsIG1lbW9yeU9mZnNldCk7XG5cbiAgICBpZiAodmFsdWVzICE9IG51bGwpIHtcbiAgICAgIHRoaXMud2FzbS5IRUFQVTguc2V0KFxuICAgICAgICAgIG5ldyBVaW50OEFycmF5KFxuICAgICAgICAgICAgICAodmFsdWVzIGFzIGJhY2tlbmRfdXRpbC5UeXBlZEFycmF5KS5idWZmZXIsXG4gICAgICAgICAgICAgICh2YWx1ZXMgYXMgYmFja2VuZF91dGlsLlR5cGVkQXJyYXkpLmJ5dGVPZmZzZXQsIG51bUJ5dGVzKSxcbiAgICAgICAgICBtZW1vcnlPZmZzZXQpO1xuICAgIH1cbiAgfVxuXG4gIGFzeW5jIHJlYWQoZGF0YUlkOiBEYXRhSWQpOiBQcm9taXNlPGJhY2tlbmRfdXRpbC5CYWNrZW5kVmFsdWVzPiB7XG4gICAgcmV0dXJuIHRoaXMucmVhZFN5bmMoZGF0YUlkKTtcbiAgfVxuXG4gIHJlYWRTeW5jKGRhdGFJZDogRGF0YUlkLCBzdGFydD86IG51bWJlciwgZW5kPzogbnVtYmVyKTpcbiAgICAgIGJhY2tlbmRfdXRpbC5CYWNrZW5kVmFsdWVzIHtcbiAgICBjb25zdCB7bWVtb3J5T2Zmc2V0LCBkdHlwZSwgc2hhcGUsIHN0cmluZ0J5dGVzfSA9XG4gICAgICAgIHRoaXMuZGF0YUlkTWFwLmdldChkYXRhSWQpO1xuICAgIGlmIChkdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICAgIC8vIFNsaWNlIGFsbCBlbGVtZW50cy5cbiAgICAgIGlmICgoc3RhcnQgPT0gbnVsbCB8fCBzdGFydCA9PT0gMCkgJiZcbiAgICAgICAgICAoZW5kID09IG51bGwgfHwgZW5kID49IHN0cmluZ0J5dGVzLmxlbmd0aCkpIHtcbiAgICAgICAgcmV0dXJuIHN0cmluZ0J5dGVzO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHN0cmluZ0J5dGVzLnNsaWNlKHN0YXJ0LCBlbmQpO1xuICAgIH1cbiAgICBzdGFydCA9IHN0YXJ0IHx8IDA7XG4gICAgZW5kID0gZW5kIHx8IHV0aWwuc2l6ZUZyb21TaGFwZShzaGFwZSk7XG4gICAgY29uc3QgYnl0ZXNQZXJFbGVtZW50ID0gdXRpbC5ieXRlc1BlckVsZW1lbnQoZHR5cGUpO1xuICAgIGNvbnN0IGJ5dGVzID0gdGhpcy53YXNtLkhFQVBVOC5zbGljZShcbiAgICAgICAgbWVtb3J5T2Zmc2V0ICsgc3RhcnQgKiBieXRlc1BlckVsZW1lbnQsXG4gICAgICAgIG1lbW9yeU9mZnNldCArIGVuZCAqIGJ5dGVzUGVyRWxlbWVudCk7XG4gICAgcmV0dXJuIHR5cGVkQXJyYXlGcm9tQnVmZmVyKGJ5dGVzLmJ1ZmZlciwgZHR5cGUpO1xuICB9XG5cbiAgLyoqXG4gICAqIERpc3Bvc2UgdGhlIG1lbW9yeSBpZiB0aGUgZGF0YUlkIGhhcyAwIHJlZkNvdW50LiBSZXR1cm4gdHJ1ZSBpZiB0aGUgbWVtb3J5XG4gICAqIGlzIHJlbGVhc2VkLCBmYWxzZSBvdGhlcndpc2UuXG4gICAqIEBwYXJhbSBkYXRhSWRcbiAgICogQG9hcmFtIGZvcmNlIE9wdGlvbmFsLCByZW1vdmUgdGhlIGRhdGEgcmVnYXJkbGVzcyBvZiByZWZDb3VudFxuICAgKi9cbiAgZGlzcG9zZURhdGEoZGF0YUlkOiBEYXRhSWQsIGZvcmNlID0gZmFsc2UpOiBib29sZWFuIHtcbiAgICBpZiAodGhpcy5kYXRhSWRNYXAuaGFzKGRhdGFJZCkpIHtcbiAgICAgIGNvbnN0IGRhdGEgPSB0aGlzLmRhdGFJZE1hcC5nZXQoZGF0YUlkKTtcbiAgICAgIGRhdGEucmVmQ291bnQtLTtcbiAgICAgIGlmICghZm9yY2UgJiYgZGF0YS5yZWZDb3VudCA+IDApIHtcbiAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgfVxuXG4gICAgICB0aGlzLndhc20uX2ZyZWUoZGF0YS5tZW1vcnlPZmZzZXQpO1xuICAgICAgdGhpcy53YXNtLnRmanMuZGlzcG9zZURhdGEoZGF0YS5pZCk7XG4gICAgICB0aGlzLmRhdGFJZE1hcC5kZWxldGUoZGF0YUlkKTtcbiAgICB9XG4gICAgcmV0dXJuIHRydWU7XG4gIH1cblxuICAvKiogUmV0dXJuIHJlZkNvdW50IG9mIGEgYFRlbnNvckRhdGFgLiAqL1xuICByZWZDb3VudChkYXRhSWQ6IERhdGFJZCk6IG51bWJlciB7XG4gICAgaWYgKHRoaXMuZGF0YUlkTWFwLmhhcyhkYXRhSWQpKSB7XG4gICAgICBjb25zdCB0ZW5zb3JEYXRhID0gdGhpcy5kYXRhSWRNYXAuZ2V0KGRhdGFJZCk7XG4gICAgICByZXR1cm4gdGVuc29yRGF0YS5yZWZDb3VudDtcbiAgICB9XG4gICAgcmV0dXJuIDA7XG4gIH1cblxuICBpbmNSZWYoZGF0YUlkOiBEYXRhSWQpIHtcbiAgICBjb25zdCBkYXRhID0gdGhpcy5kYXRhSWRNYXAuZ2V0KGRhdGFJZCk7XG4gICAgaWYgKGRhdGEgIT0gbnVsbCkge1xuICAgICAgZGF0YS5yZWZDb3VudCsrO1xuICAgIH1cbiAgfVxuXG4gIGZsb2F0UHJlY2lzaW9uKCk6IDMyIHtcbiAgICByZXR1cm4gMzI7XG4gIH1cblxuICAvLyBSZXR1cm5zIHRoZSBtZW1vcnkgb2Zmc2V0IG9mIGEgdGVuc29yLiBVc2VmdWwgZm9yIGRlYnVnZ2luZyBhbmQgdW5pdFxuICAvLyB0ZXN0aW5nLlxuICBnZXRNZW1vcnlPZmZzZXQoZGF0YUlkOiBEYXRhSWQpOiBudW1iZXIge1xuICAgIHJldHVybiB0aGlzLmRhdGFJZE1hcC5nZXQoZGF0YUlkKS5tZW1vcnlPZmZzZXQ7XG4gIH1cblxuICBkaXNwb3NlKCkge1xuICAgIHRoaXMud2FzbS50ZmpzLmRpc3Bvc2UoKTtcbiAgICBpZiAoJ1BUaHJlYWQnIGluIHRoaXMud2FzbSkge1xuICAgICAgdGhpcy53YXNtLlBUaHJlYWQudGVybWluYXRlQWxsVGhyZWFkcygpO1xuICAgIH1cbiAgICB0aGlzLndhc20gPSBudWxsO1xuICB9XG5cbiAgbWVtb3J5KCkge1xuICAgIHJldHVybiB7dW5yZWxpYWJsZTogZmFsc2V9O1xuICB9XG5cbiAgLyoqXG4gICAqIE1ha2UgYSB0ZW5zb3IgaW5mbyBmb3IgdGhlIG91dHB1dCBvZiBhbiBvcC4gSWYgYG1lbW9yeU9mZnNldGAgaXMgbm90XG4gICAqIHByZXNlbnQsIHRoaXMgbWV0aG9kIGFsbG9jYXRlcyBtZW1vcnkgb24gdGhlIFdBU00gaGVhcC4gSWYgYG1lbW9yeU9mZnNldGBcbiAgICogaXMgcHJlc2VudCwgdGhlIG1lbW9yeSB3YXMgYWxsb2NhdGVkIGVsc2V3aGVyZSAoaW4gYysrKSBhbmQgd2UganVzdCByZWNvcmRcbiAgICogdGhlIHBvaW50ZXIgd2hlcmUgdGhhdCBtZW1vcnkgbGl2ZXMuXG4gICAqL1xuICBtYWtlT3V0cHV0KHNoYXBlOiBudW1iZXJbXSwgZHR5cGU6IERhdGFUeXBlLCBtZW1vcnlPZmZzZXQ/OiBudW1iZXIpOlxuICAgICAgVGVuc29ySW5mbyB7XG4gICAgbGV0IGRhdGFJZDoge307XG4gICAgaWYgKG1lbW9yeU9mZnNldCA9PSBudWxsKSB7XG4gICAgICBkYXRhSWQgPSB0aGlzLndyaXRlKG51bGwgLyogdmFsdWVzICovLCBzaGFwZSwgZHR5cGUpO1xuICAgIH0gZWxzZSB7XG4gICAgICBjb25zdCBpZCA9IHRoaXMuZGF0YUlkTmV4dE51bWJlcisrO1xuICAgICAgZGF0YUlkID0ge2lkfTtcbiAgICAgIHRoaXMuZGF0YUlkTWFwLnNldChkYXRhSWQsIHtpZCwgbWVtb3J5T2Zmc2V0LCBzaGFwZSwgZHR5cGUsIHJlZkNvdW50OiAxfSk7XG4gICAgICBjb25zdCBzaXplID0gdXRpbC5zaXplRnJvbVNoYXBlKHNoYXBlKTtcbiAgICAgIHRoaXMud2FzbS50ZmpzLnJlZ2lzdGVyVGVuc29yKGlkLCBzaXplLCBtZW1vcnlPZmZzZXQpO1xuICAgIH1cbiAgICByZXR1cm4ge2RhdGFJZCwgc2hhcGUsIGR0eXBlfTtcbiAgfVxuXG4gIHR5cGVkQXJyYXlGcm9tSGVhcCh7c2hhcGUsIGR0eXBlLCBkYXRhSWR9OiBUZW5zb3JJbmZvKTpcbiAgICAgIGJhY2tlbmRfdXRpbC5UeXBlZEFycmF5IHtcbiAgICBjb25zdCBidWZmZXIgPSB0aGlzLndhc20uSEVBUFU4LmJ1ZmZlcjtcbiAgICBjb25zdCB7bWVtb3J5T2Zmc2V0fSA9IHRoaXMuZGF0YUlkTWFwLmdldChkYXRhSWQpO1xuICAgIGNvbnN0IHNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUoc2hhcGUpO1xuICAgIHN3aXRjaCAoZHR5cGUpIHtcbiAgICAgIGNhc2UgJ2Zsb2F0MzInOlxuICAgICAgICByZXR1cm4gbmV3IEZsb2F0MzJBcnJheShidWZmZXIsIG1lbW9yeU9mZnNldCwgc2l6ZSk7XG4gICAgICBjYXNlICdpbnQzMic6XG4gICAgICAgIHJldHVybiBuZXcgSW50MzJBcnJheShidWZmZXIsIG1lbW9yeU9mZnNldCwgc2l6ZSk7XG4gICAgICBjYXNlICdib29sJzpcbiAgICAgICAgcmV0dXJuIG5ldyBVaW50OEFycmF5KGJ1ZmZlciwgbWVtb3J5T2Zmc2V0LCBzaXplKTtcbiAgICAgIGRlZmF1bHQ6XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihgVW5rbm93biBkdHlwZSAke2R0eXBlfWApO1xuICAgIH1cbiAgfVxufVxuXG5mdW5jdGlvbiBjcmVhdGVJbnN0YW50aWF0ZVdhc21GdW5jKHBhdGg6IHN0cmluZykge1xuICAvLyB0aGlzIHdpbGwgYmUgcmVwbGFjZSBieSByb2xsdXAgcGx1Z2luIHBhdGNoV2VjaGF0V2ViQXNzZW1ibHkgaW5cbiAgLy8gbWlucHJvZ3JhbSdzIG91dHB1dC5cbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICByZXR1cm4gKGltcG9ydHM6IGFueSwgY2FsbGJhY2s6IGFueSkgPT4ge1xuICAgIHV0aWwuZmV0Y2gocGF0aCwge2NyZWRlbnRpYWxzOiAnc2FtZS1vcmlnaW4nfSkudGhlbigocmVzcG9uc2UpID0+IHtcbiAgICAgIGlmICghcmVzcG9uc2VbJ29rJ10pIHtcbiAgICAgICAgaW1wb3J0cy5lbnYuYShgZmFpbGVkIHRvIGxvYWQgd2FzbSBiaW5hcnkgZmlsZSBhdCAnJHtwYXRofSdgKTtcbiAgICAgIH1cbiAgICAgIHJlc3BvbnNlLmFycmF5QnVmZmVyKCkudGhlbihiaW5hcnkgPT4ge1xuICAgICAgICBXZWJBc3NlbWJseS5pbnN0YW50aWF0ZShiaW5hcnksIGltcG9ydHMpLnRoZW4ob3V0cHV0ID0+IHtcbiAgICAgICAgICBjYWxsYmFjayhvdXRwdXQuaW5zdGFuY2UsIG91dHB1dC5tb2R1bGUpO1xuICAgICAgICB9KTtcbiAgICAgIH0pO1xuICAgIH0pO1xuICAgIHJldHVybiB7fTtcbiAgfTtcbn1cblxuLyoqXG4gKiBSZXR1cm5zIHRoZSBwYXRoIG9mIHRoZSBXQVNNIGJpbmFyeS5cbiAqIEBwYXJhbSBzaW1kU3VwcG9ydGVkIHdoZXRoZXIgU0lNRCBpcyBzdXBwb3J0ZWRcbiAqIEBwYXJhbSB0aHJlYWRzU3VwcG9ydGVkIHdoZXRoZXIgbXVsdGl0aHJlYWRpbmcgaXMgc3VwcG9ydGVkXG4gKiBAcGFyYW0gd2FzbU1vZHVsZUZvbGRlciB0aGUgZGlyZWN0b3J5IGNvbnRhaW5pbmcgdGhlIFdBU00gYmluYXJpZXMuXG4gKi9cbmZ1bmN0aW9uIGdldFBhdGhUb1dhc21CaW5hcnkoXG4gICAgc2ltZFN1cHBvcnRlZDogYm9vbGVhbiwgdGhyZWFkc1N1cHBvcnRlZDogYm9vbGVhbixcbiAgICB3YXNtTW9kdWxlRm9sZGVyOiBzdHJpbmcpIHtcbiAgaWYgKHdhc21QYXRoICE9IG51bGwpIHtcbiAgICAvLyBJZiB3YXNtUGF0aCBpcyBkZWZpbmVkLCB0aGUgdXNlciBoYXMgc3VwcGxpZWQgYSBmdWxsIHBhdGggdG9cbiAgICAvLyB0aGUgdmFuaWxsYSAud2FzbSBiaW5hcnkuXG4gICAgcmV0dXJuIHdhc21QYXRoO1xuICB9XG5cbiAgbGV0IHBhdGg6IFdhc21CaW5hcnlOYW1lID0gJ3RmanMtYmFja2VuZC13YXNtLndhc20nO1xuICBpZiAoc2ltZFN1cHBvcnRlZCAmJiB0aHJlYWRzU3VwcG9ydGVkKSB7XG4gICAgcGF0aCA9ICd0ZmpzLWJhY2tlbmQtd2FzbS10aHJlYWRlZC1zaW1kLndhc20nO1xuICB9IGVsc2UgaWYgKHNpbWRTdXBwb3J0ZWQpIHtcbiAgICBwYXRoID0gJ3RmanMtYmFja2VuZC13YXNtLXNpbWQud2FzbSc7XG4gIH1cblxuICBpZiAod2FzbUZpbGVNYXAgIT0gbnVsbCkge1xuICAgIGlmICh3YXNtRmlsZU1hcFtwYXRoXSAhPSBudWxsKSB7XG4gICAgICByZXR1cm4gd2FzbUZpbGVNYXBbcGF0aF07XG4gICAgfVxuICB9XG5cbiAgcmV0dXJuIHdhc21Nb2R1bGVGb2xkZXIgKyBwYXRoO1xufVxuXG4vKipcbiAqIEluaXRpYWxpemVzIHRoZSB3YXNtIG1vZHVsZSBhbmQgY3JlYXRlcyB0aGUganMgPC0tPiB3YXNtIGJyaWRnZS5cbiAqXG4gKiBOT1RFOiBXZSB3cmFwIHRoZSB3YXNtIG1vZHVsZSBpbiBhIG9iamVjdCB3aXRoIHByb3BlcnR5ICd3YXNtJyBpbnN0ZWFkIG9mXG4gKiByZXR1cm5pbmcgUHJvbWlzZTxCYWNrZW5kV2FzbU1vZHVsZT4gdG8gYXZvaWQgZnJlZXppbmcgQ2hyb21lIChsYXN0IHRlc3RlZFxuICogaW4gQ2hyb21lIDc2KS5cbiAqL1xuZXhwb3J0IGFzeW5jIGZ1bmN0aW9uIGluaXQoKTogUHJvbWlzZTx7d2FzbTogQmFja2VuZFdhc21Nb2R1bGV9PiB7XG4gIGNvbnN0IFtzaW1kU3VwcG9ydGVkLCB0aHJlYWRzU3VwcG9ydGVkXSA9IGF3YWl0IFByb21pc2UuYWxsKFtcbiAgICBlbnYoKS5nZXRBc3luYygnV0FTTV9IQVNfU0lNRF9TVVBQT1JUJyksXG4gICAgZW52KCkuZ2V0QXN5bmMoJ1dBU01fSEFTX01VTFRJVEhSRUFEX1NVUFBPUlQnKVxuICBdKTtcblxuICByZXR1cm4gbmV3IFByb21pc2UoKHJlc29sdmUsIHJlamVjdCkgPT4ge1xuICAgIGNvbnN0IGZhY3RvcnlDb25maWc6IFdhc21GYWN0b3J5Q29uZmlnID0ge307XG5cbiAgICAvKipcbiAgICAgKiBUaGlzIGZ1bmN0aW9uIG92ZXJyaWRlcyB0aGUgRW1zY3JpcHRlbiBtb2R1bGUgbG9jYXRlRmlsZSB1dGlsaXR5LlxuICAgICAqIEBwYXJhbSBwYXRoIFRoZSByZWxhdGl2ZSBwYXRoIHRvIHRoZSBmaWxlIHRoYXQgbmVlZHMgdG8gYmUgbG9hZGVkLlxuICAgICAqIEBwYXJhbSBwcmVmaXggVGhlIHBhdGggdG8gdGhlIG1haW4gSmF2YVNjcmlwdCBmaWxlJ3MgZGlyZWN0b3J5LlxuICAgICAqL1xuICAgIGZhY3RvcnlDb25maWcubG9jYXRlRmlsZSA9IChwYXRoLCBwcmVmaXgpID0+IHtcbiAgICAgIGlmIChwYXRoLmVuZHNXaXRoKCcud29ya2VyLmpzJykpIHtcbiAgICAgICAgLy8gRXNjYXBlICdcXG4nIGJlY2F1c2UgQmxvYiB3aWxsIHR1cm4gaXQgaW50byBhIG5ld2xpbmUuXG4gICAgICAgIC8vIFRoZXJlIHNob3VsZCBiZSBhIHNldHRpbmcgZm9yIHRoaXMsIGJ1dCAnZW5kaW5nczogXCJuYXRpdmVcIicgZG9lc1xuICAgICAgICAvLyBub3Qgc2VlbSB0byB3b3JrLlxuICAgICAgICBjb25zdCByZXNwb25zZSA9ICh3YXNtV29ya2VyQ29udGVudHMgYXMgc3RyaW5nKS5yZXBsYWNlKC9cXG4vZywgJ1xcXFxuJyk7XG4gICAgICAgIGNvbnN0IGJsb2IgPSBuZXcgQmxvYihbcmVzcG9uc2VdLCB7dHlwZTogJ2FwcGxpY2F0aW9uL2phdmFzY3JpcHQnfSk7XG4gICAgICAgIHJldHVybiBVUkwuY3JlYXRlT2JqZWN0VVJMKGJsb2IpO1xuICAgICAgfVxuXG4gICAgICBpZiAocGF0aC5lbmRzV2l0aCgnLndhc20nKSkge1xuICAgICAgICByZXR1cm4gZ2V0UGF0aFRvV2FzbUJpbmFyeShcbiAgICAgICAgICAgIHNpbWRTdXBwb3J0ZWQgYXMgYm9vbGVhbiwgdGhyZWFkc1N1cHBvcnRlZCBhcyBib29sZWFuLFxuICAgICAgICAgICAgd2FzbVBhdGhQcmVmaXggIT0gbnVsbCA/IHdhc21QYXRoUHJlZml4IDogcHJlZml4KTtcbiAgICAgIH1cbiAgICAgIHJldHVybiBwcmVmaXggKyBwYXRoO1xuICAgIH07XG5cbiAgICAvLyBVc2UgdGhlIGluc3RhbnRpYXRlV2FzbSBvdmVycmlkZSB3aGVuIHN5c3RlbSBmZXRjaCBpcyBub3QgYXZhaWxhYmxlLlxuICAgIC8vIFJlZmVyZW5jZTpcbiAgICAvLyBodHRwczovL2dpdGh1Yi5jb20vZW1zY3JpcHRlbi1jb3JlL2Vtc2NyaXB0ZW4vYmxvYi8yYmNhMDgzY2JiZDVhNDEzM2RiNjFmYmQ3NGQwNGY3ZmVlY2ZhOTA3L3Rlc3RzL21hbnVhbF93YXNtX2luc3RhbnRpYXRlLmh0bWwjTDE3MFxuICAgIGlmIChjdXN0b21GZXRjaCkge1xuICAgICAgZmFjdG9yeUNvbmZpZy5pbnN0YW50aWF0ZVdhc20gPVxuICAgICAgICAgIGNyZWF0ZUluc3RhbnRpYXRlV2FzbUZ1bmMoZ2V0UGF0aFRvV2FzbUJpbmFyeShcbiAgICAgICAgICAgICAgc2ltZFN1cHBvcnRlZCBhcyBib29sZWFuLCB0aHJlYWRzU3VwcG9ydGVkIGFzIGJvb2xlYW4sXG4gICAgICAgICAgICAgIHdhc21QYXRoUHJlZml4ICE9IG51bGwgPyB3YXNtUGF0aFByZWZpeCA6ICcnKSk7XG4gICAgfVxuXG4gICAgbGV0IGluaXRpYWxpemVkID0gZmFsc2U7XG4gICAgZmFjdG9yeUNvbmZpZy5vbkFib3J0ID0gKCkgPT4ge1xuICAgICAgaWYgKGluaXRpYWxpemVkKSB7XG4gICAgICAgIC8vIEVtc2NyaXB0ZW4gYWxyZWFkeSBjYWxsZWQgY29uc29sZS53YXJuIHNvIG5vIG5lZWQgdG8gZG91YmxlIGxvZy5cbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuICAgICAgaWYgKGluaXRBYm9ydGVkKSB7XG4gICAgICAgIC8vIEVtc2NyaXB0ZW4gY2FsbHMgYG9uQWJvcnRgIHR3aWNlLCByZXN1bHRpbmcgaW4gZG91YmxlIGVycm9yXG4gICAgICAgIC8vIG1lc3NhZ2VzLlxuICAgICAgICByZXR1cm47XG4gICAgICB9XG4gICAgICBpbml0QWJvcnRlZCA9IHRydWU7XG4gICAgICBjb25zdCByZWplY3RNc2cgPVxuICAgICAgICAgICdNYWtlIHN1cmUgdGhlIHNlcnZlciBjYW4gc2VydmUgdGhlIGAud2FzbWAgZmlsZSByZWxhdGl2ZSB0byB0aGUgJyArXG4gICAgICAgICAgJ2J1bmRsZWQganMgZmlsZS4gRm9yIG1vcmUgZGV0YWlscyBzZWUgaHR0cHM6Ly9naXRodWIuY29tL3RlbnNvcmZsb3cvdGZqcy9ibG9iL21hc3Rlci90ZmpzLWJhY2tlbmQtd2FzbS9SRUFETUUubWQjdXNpbmctYnVuZGxlcnMnO1xuICAgICAgcmVqZWN0KHttZXNzYWdlOiByZWplY3RNc2d9KTtcbiAgICB9O1xuXG4gICAgbGV0IHdhc206IFByb21pc2U8QmFja2VuZFdhc21Nb2R1bGU+O1xuICAgIC8vIElmIGB3YXNtUGF0aGAgaGFzIGJlZW4gZGVmaW5lZCB3ZSBtdXN0IGluaXRpYWxpemUgdGhlIHZhbmlsbGEgbW9kdWxlLlxuICAgIGlmICh0aHJlYWRzU3VwcG9ydGVkICYmIHNpbWRTdXBwb3J0ZWQgJiYgd2FzbVBhdGggPT0gbnVsbCkge1xuICAgICAgZmFjdG9yeUNvbmZpZy5tYWluU2NyaXB0VXJsT3JCbG9iID0gbmV3IEJsb2IoXG4gICAgICAgICAgW2B2YXIgV2FzbUJhY2tlbmRNb2R1bGVUaHJlYWRlZFNpbWQgPSBgICtcbiAgICAgICAgICAgd2FzbUZhY3RvcnlUaHJlYWRlZFNpbWQudG9TdHJpbmcoKV0sXG4gICAgICAgICAge3R5cGU6ICd0ZXh0L2phdmFzY3JpcHQnfSk7XG4gICAgICB3YXNtID0gd2FzbUZhY3RvcnlUaHJlYWRlZFNpbWQoZmFjdG9yeUNvbmZpZyk7XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIFRoZSB3YXNtRmFjdG9yeSB3b3JrcyBmb3IgYm90aCB2YW5pbGxhIGFuZCBTSU1EIGJpbmFyaWVzLlxuICAgICAgd2FzbSA9IHdhc21GYWN0b3J5KGZhY3RvcnlDb25maWcpO1xuICAgIH1cblxuICAgIC8vIFRoZSBgd2FzbWAgcHJvbWlzZSB3aWxsIHJlc29sdmUgdG8gdGhlIFdBU00gbW9kdWxlIGNyZWF0ZWQgYnlcbiAgICAvLyB0aGUgZmFjdG9yeSwgYnV0IGl0IG1pZ2h0IGhhdmUgaGFkIGVycm9ycyBkdXJpbmcgY3JlYXRpb24uIE1vc3RcbiAgICAvLyBlcnJvcnMgYXJlIGNhdWdodCBieSB0aGUgb25BYm9ydCBjYWxsYmFjayBkZWZpbmVkIGFib3ZlLlxuICAgIC8vIEhvd2V2ZXIsIHNvbWUgZXJyb3JzLCBzdWNoIGFzIHRob3NlIG9jY3VycmluZyBmcm9tIGFcbiAgICAvLyBmYWlsZWQgZmV0Y2gsIHJlc3VsdCBpbiB0aGlzIHByb21pc2UgYmVpbmcgcmVqZWN0ZWQuIFRoZXNlIGFyZVxuICAgIC8vIGNhdWdodCBhbmQgcmUtcmVqZWN0ZWQgYmVsb3cuXG4gICAgd2FzbS50aGVuKChtb2R1bGUpID0+IHtcbiAgICAgIGluaXRpYWxpemVkID0gdHJ1ZTtcbiAgICAgIGluaXRBYm9ydGVkID0gZmFsc2U7XG5cbiAgICAgIGNvbnN0IHZvaWRSZXR1cm5UeXBlOiBzdHJpbmcgPSBudWxsO1xuICAgICAgLy8gVXNpbmcgdGhlIHRmanMgbmFtZXNwYWNlIHRvIGF2b2lkIGNvbmZsaWN0IHdpdGggZW1zY3JpcHRlbidzIEFQSS5cbiAgICAgIG1vZHVsZS50ZmpzID0ge1xuICAgICAgICBpbml0OiBtb2R1bGUuY3dyYXAoJ2luaXQnLCBudWxsLCBbXSksXG4gICAgICAgIGluaXRXaXRoVGhyZWFkc0NvdW50OlxuICAgICAgICAgICAgbW9kdWxlLmN3cmFwKCdpbml0X3dpdGhfdGhyZWFkc19jb3VudCcsIG51bGwsIFsnbnVtYmVyJ10pLFxuICAgICAgICBnZXRUaHJlYWRzQ291bnQ6IG1vZHVsZS5jd3JhcCgnZ2V0X3RocmVhZHNfY291bnQnLCAnbnVtYmVyJywgW10pLFxuICAgICAgICByZWdpc3RlclRlbnNvcjogbW9kdWxlLmN3cmFwKFxuICAgICAgICAgICAgJ3JlZ2lzdGVyX3RlbnNvcicsIG51bGwsXG4gICAgICAgICAgICBbXG4gICAgICAgICAgICAgICdudW1iZXInLCAgLy8gaWRcbiAgICAgICAgICAgICAgJ251bWJlcicsICAvLyBzaXplXG4gICAgICAgICAgICAgICdudW1iZXInLCAgLy8gbWVtb3J5T2Zmc2V0XG4gICAgICAgICAgICBdKSxcbiAgICAgICAgZGlzcG9zZURhdGE6IG1vZHVsZS5jd3JhcCgnZGlzcG9zZV9kYXRhJywgdm9pZFJldHVyblR5cGUsIFsnbnVtYmVyJ10pLFxuICAgICAgICBkaXNwb3NlOiBtb2R1bGUuY3dyYXAoJ2Rpc3Bvc2UnLCB2b2lkUmV0dXJuVHlwZSwgW10pLFxuICAgICAgfTtcblxuICAgICAgcmVzb2x2ZSh7d2FzbTogbW9kdWxlfSk7XG4gICAgfSkuY2F0Y2gocmVqZWN0KTtcbiAgfSk7XG59XG5cbmZ1bmN0aW9uIHR5cGVkQXJyYXlGcm9tQnVmZmVyKFxuICAgIGJ1ZmZlcjogQXJyYXlCdWZmZXIsIGR0eXBlOiBEYXRhVHlwZSk6IGJhY2tlbmRfdXRpbC5UeXBlZEFycmF5IHtcbiAgc3dpdGNoIChkdHlwZSkge1xuICAgIGNhc2UgJ2Zsb2F0MzInOlxuICAgICAgcmV0dXJuIG5ldyBGbG9hdDMyQXJyYXkoYnVmZmVyKTtcbiAgICBjYXNlICdpbnQzMic6XG4gICAgICByZXR1cm4gbmV3IEludDMyQXJyYXkoYnVmZmVyKTtcbiAgICBjYXNlICdib29sJzpcbiAgICAgIHJldHVybiBuZXcgVWludDhBcnJheShidWZmZXIpO1xuICAgIGRlZmF1bHQ6XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYFVua25vd24gZHR5cGUgJHtkdHlwZX1gKTtcbiAgfVxufVxuXG5jb25zdCB3YXNtQmluYXJ5TmFtZXMgPSBbXG4gICd0ZmpzLWJhY2tlbmQtd2FzbS53YXNtJywgJ3RmanMtYmFja2VuZC13YXNtLXNpbWQud2FzbScsXG4gICd0ZmpzLWJhY2tlbmQtd2FzbS10aHJlYWRlZC1zaW1kLndhc20nXG5dIGFzIGNvbnN0IDtcbnR5cGUgV2FzbUJpbmFyeU5hbWUgPSB0eXBlb2Ygd2FzbUJpbmFyeU5hbWVzW251bWJlcl07XG5cbmxldCB3YXNtUGF0aDogc3RyaW5nID0gbnVsbDtcbmxldCB3YXNtUGF0aFByZWZpeDogc3RyaW5nID0gbnVsbDtcbmxldCB3YXNtRmlsZU1hcDoge1trZXkgaW4gV2FzbUJpbmFyeU5hbWVdPzogc3RyaW5nfSA9IHt9O1xubGV0IGluaXRBYm9ydGVkID0gZmFsc2U7XG5sZXQgY3VzdG9tRmV0Y2ggPSBmYWxzZTtcblxuLyoqXG4gKiBAZGVwcmVjYXRlZCBVc2UgYHNldFdhc21QYXRoc2AgaW5zdGVhZC5cbiAqIFNldHMgdGhlIHBhdGggdG8gdGhlIGAud2FzbWAgZmlsZSB3aGljaCB3aWxsIGJlIGZldGNoZWQgd2hlbiB0aGUgd2FzbVxuICogYmFja2VuZCBpcyBpbml0aWFsaXplZC4gU2VlXG4gKiBodHRwczovL2dpdGh1Yi5jb20vdGVuc29yZmxvdy90ZmpzL2Jsb2IvbWFzdGVyL3RmanMtYmFja2VuZC13YXNtL1JFQURNRS5tZCN1c2luZy1idW5kbGVyc1xuICogZm9yIG1vcmUgZGV0YWlscy5cbiAqIEBwYXJhbSBwYXRoIHdhc20gZmlsZSBwYXRoIG9yIHVybFxuICogQHBhcmFtIHVzZVBsYXRmb3JtRmV0Y2ggb3B0aW9uYWwgYm9vbGVhbiB0byB1c2UgcGxhdGZvcm0gZmV0Y2ggdG8gZG93bmxvYWRcbiAqICAgICB0aGUgd2FzbSBmaWxlLCBkZWZhdWx0IHRvIGZhbHNlLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdFbnZpcm9ubWVudCcsIG5hbWVzcGFjZTogJ3dhc20nfVxuICovXG5leHBvcnQgZnVuY3Rpb24gc2V0V2FzbVBhdGgocGF0aDogc3RyaW5nLCB1c2VQbGF0Zm9ybUZldGNoID0gZmFsc2UpOiB2b2lkIHtcbiAgZGVwcmVjYXRpb25XYXJuKFxuICAgICAgJ3NldFdhc21QYXRoIGhhcyBiZWVuIGRlcHJlY2F0ZWQgaW4gZmF2b3Igb2Ygc2V0V2FzbVBhdGhzIGFuZCcgK1xuICAgICAgJyB3aWxsIGJlIHJlbW92ZWQgaW4gYSBmdXR1cmUgcmVsZWFzZS4nKTtcbiAgaWYgKGluaXRBYm9ydGVkKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAnVGhlIFdBU00gYmFja2VuZCB3YXMgYWxyZWFkeSBpbml0aWFsaXplZC4gTWFrZSBzdXJlIHlvdSBjYWxsICcgK1xuICAgICAgICAnYHNldFdhc21QYXRoKClgIGJlZm9yZSB5b3UgY2FsbCBgdGYuc2V0QmFja2VuZCgpYCBvciBgdGYucmVhZHkoKWAnKTtcbiAgfVxuICB3YXNtUGF0aCA9IHBhdGg7XG4gIGN1c3RvbUZldGNoID0gdXNlUGxhdGZvcm1GZXRjaDtcbn1cblxuLyoqXG4gKiBDb25maWd1cmVzIHRoZSBsb2NhdGlvbnMgb2YgdGhlIFdBU00gYmluYXJpZXMuXG4gKlxuICogYGBganNcbiAqIHNldFdhc21QYXRocyh7XG4gKiAgJ3RmanMtYmFja2VuZC13YXNtLndhc20nOiAncmVuYW1lZC53YXNtJyxcbiAqICAndGZqcy1iYWNrZW5kLXdhc20tc2ltZC53YXNtJzogJ3JlbmFtZWQtc2ltZC53YXNtJyxcbiAqICAndGZqcy1iYWNrZW5kLXdhc20tdGhyZWFkZWQtc2ltZC53YXNtJzogJ3JlbmFtZWQtdGhyZWFkZWQtc2ltZC53YXNtJ1xuICogfSk7XG4gKiB0Zi5zZXRCYWNrZW5kKCd3YXNtJyk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0gcHJlZml4T3JGaWxlTWFwIFRoaXMgY2FuIGJlIGVpdGhlciBhIHN0cmluZyBvciBvYmplY3Q6XG4gKiAgLSAoc3RyaW5nKSBUaGUgcGF0aCB0byB0aGUgZGlyZWN0b3J5IHdoZXJlIHRoZSBXQVNNIGJpbmFyaWVzIGFyZSBsb2NhdGVkLlxuICogICAgIE5vdGUgdGhhdCB0aGlzIHByZWZpeCB3aWxsIGJlIHVzZWQgdG8gbG9hZCBlYWNoIGJpbmFyeSAodmFuaWxsYSxcbiAqICAgICBTSU1ELWVuYWJsZWQsIHRocmVhZGluZy1lbmFibGVkLCBldGMuKS5cbiAqICAtIChvYmplY3QpIE1hcHBpbmcgZnJvbSBuYW1lcyBvZiBXQVNNIGJpbmFyaWVzIHRvIGN1c3RvbVxuICogICAgIGZ1bGwgcGF0aHMgc3BlY2lmeWluZyB0aGUgbG9jYXRpb25zIG9mIHRob3NlIGJpbmFyaWVzLiBUaGlzIGlzIHVzZWZ1bCBpZlxuICogICAgIHlvdXIgV0FTTSBiaW5hcmllcyBhcmUgbm90IGFsbCBsb2NhdGVkIGluIHRoZSBzYW1lIGRpcmVjdG9yeSwgb3IgaWYgeW91clxuICogICAgIFdBU00gYmluYXJpZXMgaGF2ZSBiZWVuIHJlbmFtZWQuXG4gKiBAcGFyYW0gdXNlUGxhdGZvcm1GZXRjaCBvcHRpb25hbCBib29sZWFuIHRvIHVzZSBwbGF0Zm9ybSBmZXRjaCB0byBkb3dubG9hZFxuICogICAgIHRoZSB3YXNtIGZpbGUsIGRlZmF1bHQgdG8gZmFsc2UuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0Vudmlyb25tZW50JywgbmFtZXNwYWNlOiAnd2FzbSd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBzZXRXYXNtUGF0aHMoXG4gICAgcHJlZml4T3JGaWxlTWFwOiBzdHJpbmd8e1trZXkgaW4gV2FzbUJpbmFyeU5hbWVdPzogc3RyaW5nfSxcbiAgICB1c2VQbGF0Zm9ybUZldGNoID0gZmFsc2UpOiB2b2lkIHtcbiAgaWYgKGluaXRBYm9ydGVkKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAnVGhlIFdBU00gYmFja2VuZCB3YXMgYWxyZWFkeSBpbml0aWFsaXplZC4gTWFrZSBzdXJlIHlvdSBjYWxsICcgK1xuICAgICAgICAnYHNldFdhc21QYXRocygpYCBiZWZvcmUgeW91IGNhbGwgYHRmLnNldEJhY2tlbmQoKWAgb3IgJyArXG4gICAgICAgICdgdGYucmVhZHkoKWAnKTtcbiAgfVxuXG4gIGlmICh0eXBlb2YgcHJlZml4T3JGaWxlTWFwID09PSAnc3RyaW5nJykge1xuICAgIHdhc21QYXRoUHJlZml4ID0gcHJlZml4T3JGaWxlTWFwO1xuICB9IGVsc2Uge1xuICAgIHdhc21GaWxlTWFwID0gcHJlZml4T3JGaWxlTWFwO1xuICAgIGNvbnN0IG1pc3NpbmdQYXRocyA9XG4gICAgICAgIHdhc21CaW5hcnlOYW1lcy5maWx0ZXIobmFtZSA9PiB3YXNtRmlsZU1hcFtuYW1lXSA9PSBudWxsKTtcbiAgICBpZiAobWlzc2luZ1BhdGhzLmxlbmd0aCA+IDApIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgVGhlcmUgd2VyZSBubyBlbnRyaWVzIGZvdW5kIGZvciB0aGUgZm9sbG93aW5nIGJpbmFyaWVzOiBgICtcbiAgICAgICAgICBgJHttaXNzaW5nUGF0aHMuam9pbignLCcpfS4gUGxlYXNlIGVpdGhlciBjYWxsIHNldFdhc21QYXRocyB3aXRoIGEgYCArXG4gICAgICAgICAgYG1hcCBwcm92aWRpbmcgYSBwYXRoIGZvciBlYWNoIGJpbmFyeSwgb3Igd2l0aCBhIHN0cmluZyBpbmRpY2F0aW5nIGAgK1xuICAgICAgICAgIGB0aGUgZGlyZWN0b3J5IHdoZXJlIGFsbCB0aGUgYmluYXJpZXMgY2FuIGJlIGZvdW5kLmApO1xuICAgIH1cbiAgfVxuXG4gIGN1c3RvbUZldGNoID0gdXNlUGxhdGZvcm1GZXRjaDtcbn1cblxuLyoqIFVzZWQgaW4gdW5pdCB0ZXN0cy4gKi9cbmV4cG9ydCBmdW5jdGlvbiByZXNldFdhc21QYXRoKCk6IHZvaWQge1xuICB3YXNtUGF0aCA9IG51bGw7XG4gIHdhc21QYXRoUHJlZml4ID0gbnVsbDtcbiAgd2FzbUZpbGVNYXAgPSB7fTtcbiAgY3VzdG9tRmV0Y2ggPSBmYWxzZTtcbiAgaW5pdEFib3J0ZWQgPSBmYWxzZTtcbn1cblxubGV0IHRocmVhZHNDb3VudCA9IC0xO1xubGV0IGFjdHVhbFRocmVhZHNDb3VudCA9IC0xO1xuXG4vKipcbiAqIFNldHMgdGhlIG51bWJlciBvZiB0aHJlYWRzIHRoYXQgd2lsbCBiZSB1c2VkIGJ5IFhOTlBBQ0sgdG8gY3JlYXRlXG4gKiB0aHJlYWRwb29sIChkZWZhdWx0IHRvIHRoZSBudW1iZXIgb2YgbG9naWNhbCBDUFUgY29yZXMpLlxuICpcbiAqIFRoaXMgbXVzdCBiZSBjYWxsZWQgYmVmb3JlIGNhbGxpbmcgYHRmLnNldEJhY2tlbmQoJ3dhc20nKWAuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBzZXRUaHJlYWRzQ291bnQobnVtVGhyZWFkczogbnVtYmVyKSB7XG4gIHRocmVhZHNDb3VudCA9IG51bVRocmVhZHM7XG59XG5cbi8qKlxuICogR2V0cyB0aGUgYWN0dWFsIHRocmVhZHMgY291bnQgdGhhdCBpcyB1c2VkIGJ5IFhOTlBBQ0suXG4gKlxuICogSXQgaXMgc2V0IGFmdGVyIHRoZSBiYWNrZW5kIGlzIGludGlhbGl6ZWQuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBnZXRUaHJlYWRzQ291bnQoKTogbnVtYmVyIHtcbiAgaWYgKGFjdHVhbFRocmVhZHNDb3VudCA9PT0gLTEpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYFdBU00gYmFja2VuZCBub3QgaW5pdGlhbGl6ZWQuYCk7XG4gIH1cbiAgcmV0dXJuIGFjdHVhbFRocmVhZHNDb3VudDtcbn1cbiJdfQ==