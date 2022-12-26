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
import { util } from '@tensorflow/tfjs-core';
import { CppDType } from './types';
export function createUnaryKernelConfig(kernelName, outType) {
    let wasmFunc;
    function setupFunc(backend) {
        wasmFunc = backend.wasm.cwrap(kernelName, null /* void */, [
            'number',
            'number',
            'number',
        ]);
    }
    function kernelFunc(args) {
        const { backend, inputs: { x } } = args;
        const xId = backend.dataIdMap.get(x.dataId).id;
        const out = backend.makeOutput(x.shape, outType || x.dtype);
        const outId = backend.dataIdMap.get(out.dataId).id;
        // Short-circuit zero-sized tensors.
        if (util.sizeFromShape(out.shape) === 0) {
            return out;
        }
        wasmFunc(xId, CppDType[x.dtype], outId);
        return out;
    }
    return { kernelName, backendName: 'wasm', setupFunc, kernelFunc };
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidW5hcnlfa2VybmVsLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdhc20vc3JjL2tlcm5lbHMvdW5hcnlfa2VybmVsLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBa0QsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFJNUYsT0FBTyxFQUFDLFFBQVEsRUFBQyxNQUFNLFNBQVMsQ0FBQztBQUVqQyxNQUFNLFVBQVUsdUJBQXVCLENBQ25DLFVBQWtCLEVBQUUsT0FBa0I7SUFDeEMsSUFBSSxRQUE2RCxDQUFDO0lBRWxFLFNBQVMsU0FBUyxDQUFDLE9BQW9CO1FBQ3JDLFFBQVEsR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxVQUFVLEVBQUUsSUFBSSxDQUFDLFVBQVUsRUFBRTtZQUN6RCxRQUFRO1lBQ1IsUUFBUTtZQUNSLFFBQVE7U0FDVCxDQUFDLENBQUM7SUFDTCxDQUFDO0lBRUQsU0FBUyxVQUFVLENBQUMsSUFBaUQ7UUFFbkUsTUFBTSxFQUFDLE9BQU8sRUFBRSxNQUFNLEVBQUUsRUFBQyxDQUFDLEVBQUMsRUFBQyxHQUFHLElBQUksQ0FBQztRQUNwQyxNQUFNLEdBQUcsR0FBRyxPQUFPLENBQUMsU0FBUyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsTUFBTSxDQUFDLENBQUMsRUFBRSxDQUFDO1FBQy9DLE1BQU0sR0FBRyxHQUFHLE9BQU8sQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDLEtBQUssRUFBRSxPQUFPLElBQUksQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQzVELE1BQU0sS0FBSyxHQUFHLE9BQU8sQ0FBQyxTQUFTLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxFQUFFLENBQUM7UUFFbkQsb0NBQW9DO1FBQ3BDLElBQUksSUFBSSxDQUFDLGFBQWEsQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLEtBQUssQ0FBQyxFQUFFO1lBQ3ZDLE9BQU8sR0FBRyxDQUFDO1NBQ1o7UUFFRCxRQUFRLENBQUMsR0FBRyxFQUFFLFFBQVEsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLEVBQUUsS0FBSyxDQUFDLENBQUM7UUFDeEMsT0FBTyxHQUFHLENBQUM7SUFDYixDQUFDO0lBRUQsT0FBTyxFQUFDLFVBQVUsRUFBRSxXQUFXLEVBQUUsTUFBTSxFQUFFLFNBQVMsRUFBRSxVQUFVLEVBQUMsQ0FBQztBQUNsRSxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTkgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0RhdGFUeXBlLCBLZXJuZWxDb25maWcsIFRlbnNvckluZm8sIFVuYXJ5SW5wdXRzLCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge0JhY2tlbmRXYXNtfSBmcm9tICcuLi9iYWNrZW5kX3dhc20nO1xuXG5pbXBvcnQge0NwcERUeXBlfSBmcm9tICcuL3R5cGVzJztcblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVVuYXJ5S2VybmVsQ29uZmlnKFxuICAgIGtlcm5lbE5hbWU6IHN0cmluZywgb3V0VHlwZT86IERhdGFUeXBlKTogS2VybmVsQ29uZmlnIHtcbiAgbGV0IHdhc21GdW5jOiAoeElkOiBudW1iZXIsIGR0eXBlOiBudW1iZXIsIG91dElkOiBudW1iZXIpID0+IHZvaWQ7XG5cbiAgZnVuY3Rpb24gc2V0dXBGdW5jKGJhY2tlbmQ6IEJhY2tlbmRXYXNtKTogdm9pZCB7XG4gICAgd2FzbUZ1bmMgPSBiYWNrZW5kLndhc20uY3dyYXAoa2VybmVsTmFtZSwgbnVsbCAvKiB2b2lkICovLCBbXG4gICAgICAnbnVtYmVyJywgIC8vIHhfaWRcbiAgICAgICdudW1iZXInLCAgLy8gZHR5cGVcbiAgICAgICdudW1iZXInLCAgLy8gb3V0X2lkXG4gICAgXSk7XG4gIH1cblxuICBmdW5jdGlvbiBrZXJuZWxGdW5jKGFyZ3M6IHtiYWNrZW5kOiBCYWNrZW5kV2FzbSwgaW5wdXRzOiBVbmFyeUlucHV0c30pOlxuICAgICAgVGVuc29ySW5mbyB7XG4gICAgY29uc3Qge2JhY2tlbmQsIGlucHV0czoge3h9fSA9IGFyZ3M7XG4gICAgY29uc3QgeElkID0gYmFja2VuZC5kYXRhSWRNYXAuZ2V0KHguZGF0YUlkKS5pZDtcbiAgICBjb25zdCBvdXQgPSBiYWNrZW5kLm1ha2VPdXRwdXQoeC5zaGFwZSwgb3V0VHlwZSB8fCB4LmR0eXBlKTtcbiAgICBjb25zdCBvdXRJZCA9IGJhY2tlbmQuZGF0YUlkTWFwLmdldChvdXQuZGF0YUlkKS5pZDtcblxuICAgIC8vIFNob3J0LWNpcmN1aXQgemVyby1zaXplZCB0ZW5zb3JzLlxuICAgIGlmICh1dGlsLnNpemVGcm9tU2hhcGUob3V0LnNoYXBlKSA9PT0gMCkge1xuICAgICAgcmV0dXJuIG91dDtcbiAgICB9XG5cbiAgICB3YXNtRnVuYyh4SWQsIENwcERUeXBlW3guZHR5cGVdLCBvdXRJZCk7XG4gICAgcmV0dXJuIG91dDtcbiAgfVxuXG4gIHJldHVybiB7a2VybmVsTmFtZSwgYmFja2VuZE5hbWU6ICd3YXNtJywgc2V0dXBGdW5jLCBrZXJuZWxGdW5jfTtcbn1cbiJdfQ==