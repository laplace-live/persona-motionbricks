// Loads the model and clips and wires the locomotion loop and the retargeter.

import { join } from 'node:path'

import { MANIFEST_FILE, modelFile } from './assets.ts'
import { canonicalWindows, type Gear, Locomotion, Rig, type TargetWindow } from './locomotion.ts'
import { SomaModel } from './model.ts'
import { jumps } from './jump.ts'
import { type RetargetOptions, Retargeter } from './retarget.ts'
import { type Clip, forward, loadBvh, type Skeleton, samplePose } from './skeleton.ts'

/** Each gear's clip, and the times (seconds) of its target windows, picked from steady stretches. */
const CLIPS: Record<Gear, { file: string; times: number[] }> = {
  idle: { file: 'idle_loop_001__A268.bvh', times: [1, 2.5, 4, 5.5, 7, 8.5] },
  walk: { file: 'Neutral_walk_forward_002__A057.bvh', times: [2.5] },
  jog: { file: 'neutral_jog_ff_180_R_001__A534.bvh', times: [1.8] },
  run: { file: 'run_loop_180_R_003__A326.bvh', times: [0.5] },
}

export interface Demo {
  skeleton: Skeleton
  model: SomaModel
  locomotion: Locomotion
  retargeter: Retargeter
}

/** `options` picks the FP32 model, and how the retargeter fits the motion to the avatar. */
export async function createDemo(assets: string, options: { fp32?: boolean } & RetargetOptions = {}): Promise<Demo> {
  const model = await SomaModel.load(join(assets, modelFile(options.fp32)), join(assets, MANIFEST_FILE))

  const { skeleton, clip: idleClip } = loadBvh(join(assets, 'motions', CLIPS.idle.file))
  const load = (file: string): Clip => {
    const loaded = loadBvh(join(assets, 'motions', file))
    if (loaded.skeleton.names.join() !== skeleton.names.join()) throw new Error(`${file}: different skeleton`)
    return loaded.clip
  }

  // A neutral standing frame: the hand pose the model doesn't drive, and the retargeting T-pose basis.
  const standing = samplePose(idleClip.frames, idleClip.fps, 1)
  const rig = new Rig(skeleton, model.joints, standing)
  const windowsFor = (gear: Gear): TargetWindow[] => {
    const clip = gear === 'idle' ? idleClip : load(CLIPS[gear].file)
    return CLIPS[gear].times.flatMap(t => canonicalWindows(rig, clip, [t]))
  }
  const windows: Record<Gear, TargetWindow[]> = {
    idle: windowsFor('idle'),
    walk: windowsFor('walk'),
    jog: windowsFor('jog'),
    run: windowsFor('run'),
  }
  // Start as if the character had been standing at the origin, facing the camera (+Z).
  const start = windows.idle[0].poses[3]
  return {
    skeleton,
    model,
    locomotion: new Locomotion(model, rig, windows, jumps(rig, load), start),
    retargeter: new Retargeter(skeleton, forward(skeleton, standing), options),
  }
}
