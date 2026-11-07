import assert from 'node:assert/strict'
import { test } from 'bun:test'

import { dot, normalize, qmul, qrotate, type Quat, sub, type Vec3 } from '../src/math.ts'
import { BONE_PARENTS, BONES, LEVEL, Retargeter, toUnityQuat, type VmcFrame } from '../src/retarget.ts'
import { forward, jointIndex, loadBvh, samplePose } from '../src/skeleton.ts'
import { ASSETS, missingAssets } from './assets.ts'

/** Compose the sent locals back into world deltas, as a receiver applying them to a T-pose would. */
function deltasOf({ bones }: VmcFrame): Quat[] {
  const deltas: Quat[] = []
  bones.forEach(([, local], i) => {
    const parent = BONE_PARENTS[i]
    const q = toUnityQuat(local) // the mirror is its own inverse
    deltas.push(parent < 0 ? q : qmul(deltas[parent], q))
  })
  return deltas
}

const degreesBetween = (a: Vec3, b: Vec3): number => (Math.acos(Math.min(1, dot(a, b))) * 180) / Math.PI

function load() {
  const { skeleton, clip: idle } = loadBvh(`${ASSETS}motions/idle_loop_001__A268.bvh`)
  const { clip: walk } = loadBvh(`${ASSETS}motions/Neutral_walk_forward_002__A057.bvh`)
  const standing = forward(skeleton, samplePose(idle.frames, idle.fps, 1))
  return {
    skeleton,
    retargeter: (arms: number) => new Retargeter(skeleton, standing, { arms }),
    walkAt: (time: number) => forward(skeleton, samplePose(walk.frames, walk.fps, time)),
  }
}

test.skipIf(missingAssets)('VMC bones rebuild SOMA limb directions from a T-pose', () => {
  const { skeleton, retargeter, walkAt } = load()
  const exact = retargeter(0)
  const joint = (name: string): number => jointIndex(skeleton, name)

  for (const time of [2, 2.3, 2.6, 4]) {
    const world = walkAt(time)
    const deltas = deltasOf(exact.frame(world))
    BONES.forEach(({ bone, soma, swing }, i) => {
      if (!swing || swing[1] === LEVEL) return
      const [child, dir] = swing
      const want = normalize(sub(world.pos[joint(child)], world.pos[joint(soma)]))
      const degrees = degreesBetween(qrotate(deltas[i], dir), want)
      assert.ok(degrees < 0.5, `${bone} at ${time} s is off by ${degrees.toFixed(2)}°`)
    })
  }
})

test.skipIf(missingAssets)('the arm correction opens both arms outward, more at a run', () => {
  const { retargeter, walkAt } = load()
  const world = walkAt(2.3)
  const plain = deltasOf(retargeter(0).frame(world))
  for (const [pace, least] of [
    [0, 10],
    [3.5, 13],
  ]) {
    const opened = deltasOf(retargeter(1).frame(world, pace))
    for (const side of ['Left', 'Right']) {
      const index = BONES.findIndex(b => b.bone === `${side}UpperArm`)
      const along: Vec3 = side === 'Left' ? [1, 0, 0] : [-1, 0, 0]
      const before = qrotate(plain[index], along)
      const after = qrotate(opened[index], along)
      const outward = side === 'Left' ? after[0] - before[0] : before[0] - after[0]
      assert.ok(outward > 0, `${side} arm moved inward at ${pace} m/s`)
      assert.ok(degreesBetween(before, after) >= least, `${side} arm opened under ${least}° at ${pace} m/s`)
    }
  }
})
