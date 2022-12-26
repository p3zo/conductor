import { IPreprocInfo } from './preproc_comp';
import { ITrackSource } from '../track_source';
import { IModelInput } from '../model/base';
export declare function CreateHtmlCanvasBasedPreprocCb(originalWidth: number, originalHeight: number, resizeWidth: number, resizeHeight: number): (trackSource: ITrackSource, preprocInfo: IPreprocInfo) => Promise<IModelInput>;
