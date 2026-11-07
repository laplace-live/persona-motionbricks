import { existsSync } from 'node:fs'

import { ASSETS_DIR, modelFile } from '../src/assets.ts'

export const ASSETS = ASSETS_DIR

/** Tests that need NVIDIA's files skip until `bun run fetch-assets` has run. */
export const missingAssets = !existsSync(`${ASSETS}${modelFile()}`)

/** Loading the model and generating takes a few seconds on a slow CPU. */
export const MODEL_TIMEOUT_MS = 60_000
