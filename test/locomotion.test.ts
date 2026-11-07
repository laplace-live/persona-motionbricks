import assert from 'node:assert/strict'
import { test } from 'bun:test'

import { createDemo } from '../src/demo.ts'
import { type Command, type Locomotion, UPDATE_HZ } from '../src/locomotion.ts'
import { forward, jointIndex } from '../src/skeleton.ts'
import { ASSETS, MODEL_TIMEOUT_MS, missingAssets } from './assets.ts'

async function run(locomotion: Locomotion, command: Command, seconds: number): Promise<void> {
  for (let i = 0; i < seconds * UPDATE_HZ; i++) await locomotion.step(command)
}

test.skipIf(missingAssets)(
  'walking steers the character, and it stops when told',
  async () => {
    const { locomotion } = await createDemo(ASSETS)

    await run(locomotion, { gear: 'walk', speed: 1.3, dir: [1, 0, 0] }, 3)
    const walked = locomotion.where()
    assert.ok(walked.at[0] > 2 && walked.at[0] < 4.5, `walked to x = ${walked.at[0].toFixed(2)} m`)
    assert.ok(Math.abs(walked.at[2]) < 0.5, `drifted to z = ${walked.at[2].toFixed(2)} m`)
    const facing = (walked.heading * 180) / Math.PI
    assert.ok(Math.abs(facing - 90) < 25, `faces ${facing.toFixed(0)}°, not +X`)

    await run(locomotion, { gear: 'idle', speed: 0, dir: null }, 2.5)
    const settled = locomotion.where().at
    await run(locomotion, { gear: 'idle', speed: 0, dir: null }, 0.5)
    const drift = Math.hypot(locomotion.where().at[0] - settled[0], locomotion.where().at[2] - settled[2])
    assert.ok(drift < 0.1, `still moving ${drift.toFixed(2)} m per half second after stopping`)
    for (const q of locomotion.current.local) assert.ok(q.every(Number.isFinite))
    assert.ok(locomotion.lastGeneration && locomotion.lastGeneration.frames.length >= 26)
  },
  MODEL_TIMEOUT_MS
)

test.skipIf(missingAssets)(
  'a jump leaves the ground, lands, and walking resumes',
  async () => {
    const { locomotion, skeleton } = await createDemo(ASSETS)
    const walk: Command = { gear: 'walk', speed: 1.3, dir: [1, 0, 0] }
    await run(locomotion, walk, 2)
    const feet = ['LeftFoot', 'LeftToeBase', 'RightFoot', 'RightToeBase'].map(n => jointIndex(skeleton, n))

    await locomotion.step({ ...walk, jump: true })
    assert.ok(locomotion.jumping)
    let clearance = 0
    let landedAfter = -1
    for (let i = 1; i < 4 * UPDATE_HZ && landedAfter < 0; i++) {
      await locomotion.step(walk)
      const world = forward(skeleton, locomotion.current)
      clearance = Math.max(clearance, Math.min(...feet.map(j => world.pos[j][1])))
      if (!locomotion.jumping) landedAfter = i / UPDATE_HZ
    }
    assert.ok(clearance > 0.5, `both feet only rose ${clearance.toFixed(2)} m`)
    assert.ok(landedAfter > 0.5 && landedAfter < 3, `the jump took ${landedAfter.toFixed(2)} s`)

    const before = locomotion.where().at[0]
    await run(locomotion, walk, 1)
    assert.ok(locomotion.where().at[0] - before > 0.8, 'walking did not resume after landing')
  },
  MODEL_TIMEOUT_MS
)
