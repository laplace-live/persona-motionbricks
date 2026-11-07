// Where `bun run fetch-assets` puts NVIDIA's files, and the names the demo loads them by. Imports
// nothing heavy, so the fetch script and the tests can share it.

import { fileURLToPath } from 'node:url'

/** The repository's assets/ directory, with a trailing separator. */
export const ASSETS_DIR = fileURLToPath(new URL('../assets/', import.meta.url))
/** The model's joint manifest. */
export const MANIFEST_FILE = 'soma_n1.json'
export const modelFile = (fp32 = false): string => (fp32 ? 'mmm_v6_soma_fp32.onnx' : 'mmm_v6_soma_fp16.onnx')
