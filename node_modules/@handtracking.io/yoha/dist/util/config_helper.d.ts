/**
 * Take src and trg configs and recursively checks whether all fields from
 * src are contained in trg. Any time that's not the case the value from src will
 * be used to populate the field in trg.
 *
 * If trg is null or undefined a copy of src is returned.
 *
 * @param src Source config.
 * @param trg Target config.
 *
 * Note: This function doesn't modify any arguments in place but returns a copy.
 */
export declare function ApplyConfigDefaults(src: any, trg: any): any;
