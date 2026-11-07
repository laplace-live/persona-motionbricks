import assert from 'node:assert/strict'
import { test } from 'bun:test'

import { MANIFEST_FILE, modelFile } from '../src/assets.ts'
import { ALLOWED_TOKENS, Rig, canonicalWindows } from '../src/locomotion.ts'
import { add, headingDir, length, qyaw, scale, sub } from '../src/math.ts'
import { MODEL_FPS, SomaModel } from '../src/model.ts'
import { loadBvh, moveRoot, samplePose } from '../src/skeleton.ts'
import { ASSETS, MODEL_TIMEOUT_MS, missingAssets } from './assets.ts'

test.skipIf(missingAssets)(
  'one generation walks from the context to a target placed ahead',
  async () => {
    const model = await SomaModel.load(`${ASSETS}${modelFile()}`, `${ASSETS}${MANIFEST_FILE}`)
    const { skeleton, clip: idle } = loadBvh(`${ASSETS}motions/idle_loop_001__A268.bvh`)
    const { clip: walk } = loadBvh(`${ASSETS}motions/Neutral_walk_forward_002__A057.bvh`)
    const rig = new Rig(skeleton, model.joints, samplePose(idle.frames, idle.fps, 1))

    const context = [0, 1, 2, 3].map(k => samplePose(idle.frames, idle.fps, 2 + k / MODEL_FPS))
    const { at, heading } = rig.where(context[3])
    const destination = add(at, scale(headingDir(heading), 1.5))
    const [window] = canonicalWindows(rig, walk, [2.5])
    const targets = window.poses.map(p => moveRoot(p, qyaw(heading), destination))
    const asModel = (p: (typeof context)[number]) => rig.modelPose(p)

    const g = await model.generate(context.map(asModel), targets.map(asModel), ALLOWED_TOKENS)
    assert.ok(ALLOWED_TOKENS.includes(g.tokens), `chose ${g.tokens} tokens`)
    assert.equal(g.frames.length, 4 * g.tokens - 6)
    for (const f of g.frames) assert.ok([...f.root, ...f.rot.flat()].every(Number.isFinite))
    const startError = length(sub(g.frames[0].root, context[3].root))
    const endError = length(sub(g.frames[g.frames.length - 1].root, targets[0].root))
    assert.ok(startError < 0.02, `starts ${(startError * 100).toFixed(1)} cm from the context`)
    assert.ok(endError < 0.03, `ends ${(endError * 100).toFixed(1)} cm from the target`)
    for (const c of g.contacts) assert.ok(c.length === 4 && c.every(v => v >= 0 && v <= 1))
  },
  MODEL_TIMEOUT_MS
)
