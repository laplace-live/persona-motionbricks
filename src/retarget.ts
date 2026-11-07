// SOMA → humanoid retargeting for VMC. A VMC bone rotation is the bone's local rotation away from
// a T-pose with identity bone frames, in Unity's left-handed space, as VSeeFace and Warudo send.
//
// SOMA's joint frames point +X down the bone, so each joint gets a T-pose reference rotation R
// built from a neutral standing frame: limbs, fingers and spine are swung onto their T-pose
// directions, feet are turned straight ahead, and the head and shoulders keep their standing orientation.
// A frame then maps by world deltas G·R⁻¹, made parent-relative along the humanoid chain.

import { normalize, type Quat, qaxisAngle, qfromTo, qinv, qmul, qrotate, qyaw, scale, sub, type Vec3 } from './math.ts'
import { jointIndex, localFromWorld, poseHeading, type Skeleton, type WorldPose } from './skeleton.ts'

const UP: Vec3 = [0, 1, 0]
const DOWN: Vec3 = [0, -1, 0]
/** Facing +Z, the character's left is +X. */
const LEFT: Vec3 = [1, 0, 0]
const RIGHT: Vec3 = [-1, 0, 0]

/** Straight ahead at the standing pitch: feet keep their tilt but lose the stance's toe-out. */
export const LEVEL = 'level'

interface BoneMap {
  /** Unity HumanBodyBones name, as VMC sends it. */
  bone: string
  soma: string
  parent: string | null
  /** SOMA joint the bone points at, and its T-pose direction; null keeps the standing orientation. */
  swing: [child: string, dir: Vec3 | typeof LEVEL] | null
}

/**
 * The model has no fingers, so they keep the mocap actor's relaxed hand from the standing frame.
 * VRM T-poses lay fingers along the hand and point thumbs about 45° forward, slightly down.
 * Unity's thumb Proximal/Intermediate/Distal are SOMA's Thumb1–3; the fingers' are SOMA's 2–4.
 */
function fingers(side: 'Left' | 'Right'): BoneMap[] {
  const sign = side === 'Left' ? 1 : -1
  const along: Vec3 = [sign, 0, 0]
  const thumb = normalize([sign * 0.72, -0.1, 0.68])
  const bones: BoneMap[] = []
  for (const [unity, soma] of [
    ['Thumb', 'Thumb'],
    ['Index', 'Index'],
    ['Middle', 'Middle'],
    ['Ring', 'Ring'],
    ['Little', 'Pinky'],
  ]) {
    const joints = soma === 'Thumb' ? [1, 2, 3, 'End'] : [2, 3, 4, 'End']
    ;['Proximal', 'Intermediate', 'Distal'].forEach((segment, i) => {
      bones.push({
        bone: `${side}${unity}${segment}`,
        soma: `${side}Hand${soma}${joints[i]}`,
        parent: i === 0 ? `${side}Hand` : `${side}${unity}${['Proximal', 'Intermediate'][i - 1]}`,
        swing: [`${side}Hand${soma}${joints[i + 1]}`, soma === 'Thumb' ? thumb : along],
      })
    })
  }
  return bones
}

export const BONES: BoneMap[] = [
  { bone: 'Hips', soma: 'Hips', parent: null, swing: ['Spine1', UP] },
  { bone: 'Spine', soma: 'Spine1', parent: 'Hips', swing: ['Spine2', UP] },
  { bone: 'Chest', soma: 'Spine2', parent: 'Spine', swing: ['Chest', UP] },
  { bone: 'UpperChest', soma: 'Chest', parent: 'Chest', swing: ['Neck1', UP] },
  { bone: 'Neck', soma: 'Neck1', parent: 'UpperChest', swing: ['Neck2', UP] },
  { bone: 'Head', soma: 'Head', parent: 'Neck', swing: null },
  { bone: 'LeftShoulder', soma: 'LeftShoulder', parent: 'UpperChest', swing: null },
  { bone: 'LeftUpperArm', soma: 'LeftArm', parent: 'LeftShoulder', swing: ['LeftForeArm', LEFT] },
  { bone: 'LeftLowerArm', soma: 'LeftForeArm', parent: 'LeftUpperArm', swing: ['LeftHand', LEFT] },
  { bone: 'LeftHand', soma: 'LeftHand', parent: 'LeftLowerArm', swing: ['LeftHandMiddle2', LEFT] },
  { bone: 'RightShoulder', soma: 'RightShoulder', parent: 'UpperChest', swing: null },
  { bone: 'RightUpperArm', soma: 'RightArm', parent: 'RightShoulder', swing: ['RightForeArm', RIGHT] },
  { bone: 'RightLowerArm', soma: 'RightForeArm', parent: 'RightUpperArm', swing: ['RightHand', RIGHT] },
  { bone: 'RightHand', soma: 'RightHand', parent: 'RightLowerArm', swing: ['RightHandMiddle2', RIGHT] },
  { bone: 'LeftUpperLeg', soma: 'LeftLeg', parent: 'Hips', swing: ['LeftShin', DOWN] },
  { bone: 'LeftLowerLeg', soma: 'LeftShin', parent: 'LeftUpperLeg', swing: ['LeftFoot', DOWN] },
  { bone: 'LeftFoot', soma: 'LeftFoot', parent: 'LeftLowerLeg', swing: ['LeftToeBase', LEVEL] },
  { bone: 'LeftToes', soma: 'LeftToeBase', parent: 'LeftFoot', swing: ['LeftToeEnd', LEVEL] },
  { bone: 'RightUpperLeg', soma: 'RightLeg', parent: 'Hips', swing: ['RightShin', DOWN] },
  { bone: 'RightLowerLeg', soma: 'RightShin', parent: 'RightUpperLeg', swing: ['RightFoot', DOWN] },
  { bone: 'RightFoot', soma: 'RightFoot', parent: 'RightLowerLeg', swing: ['RightToeBase', LEVEL] },
  { bone: 'RightToes', soma: 'RightToeBase', parent: 'RightFoot', swing: ['RightToeEnd', LEVEL] },
  ...fingers('Left'),
  ...fingers('Right'),
]

/** Each bone's parent as an index into BONES; -1 for Hips. */
export const BONE_PARENTS = BONES.map(b => BONES.findIndex(p => p.bone === b.parent))

export interface VmcFrame {
  /** Hips position, Unity space (metres), scaled to the avatar. */
  hips: Vec3
  /** Local rotation per bone, Unity space, in BONES order. */
  bones: Array<[bone: string, rotation: Quat]>
}

/** Right-handed SOMA (Y up, facing +Z) → Unity's left-handed space: mirror X. */
export const toUnityVec = (v: Vec3): Vec3 => [-v[0], v[1], v[2]]
export const toUnityQuat = (q: Quat): Quat => [q[0], -q[1], -q[2], q[3]]

/**
 * Stylised avatars have wider chests and hips than the mocap actor, so arm swings that clear a
 * real body cut into theirs. Each upper arm opens away from the body and turns outward about its
 * own length, by these degrees: a little while walking (hands clear the hips and skirt), more from
 * a jog up (forearms swing ahead instead of across the chest). Tuned on five VRoid-style avatars.
 */
const ARMS_AT_WALK = { open: 12, twist: 10 }
const ARMS_AT_RUN = { open: 15, twist: 35 }
/** Ground speeds (m/s) between which the correction blends from walk to run. */
const WALK_PACE = 1.4
const RUN_PACE = 2.4

export interface RetargetOptions {
  /** Scales the arm correction (default 1): 0 sends the mocap arms unchanged. */
  arms?: number
  /** The avatar's hips height ÷ the model's 0.99 m (default 1), so its travel matches its legs. */
  scale?: number
}

export class Retargeter {
  private readonly joints: number[]
  private readonly references: Quat[]
  private readonly hips: number
  private readonly arms: number
  private readonly scale: number

  /** `standing` is a neutral standing frame of the same skeleton; its facing doesn't matter. */
  constructor(skeleton: Skeleton, standing: WorldPose, options: RetargetOptions = {}) {
    this.arms = options.arms ?? 1
    this.scale = options.scale ?? 1
    const joint = (name: string): number => jointIndex(skeleton, name)
    this.hips = joint('Hips')
    this.joints = BONES.map(b => joint(b.soma))
    const face = qyaw(-poseHeading(skeleton, standing))
    this.references = BONES.map((b, i) => {
      const rest = qmul(face, standing.rot[this.joints[i]])
      if (!b.swing) return rest
      const [child, dir] = b.swing
      const along = normalize(qrotate(face, sub(standing.pos[joint(child)], standing.pos[this.joints[i]])))
      const to = dir === LEVEL ? normalize([0, along[1], Math.hypot(along[0], along[2])]) : dir
      return qmul(qfromTo(along, to), rest)
    })
  }

  /** World-space bone deltas from the T-pose, in SOMA space. */
  private deltas(world: WorldPose): Quat[] {
    return this.joints.map((j, i) => qmul(world.rot[j], qinv(this.references[i])))
  }

  /** `pace` is the character's ground speed (m/s), which sets how far the arms open. */
  frame(world: WorldPose, pace = 0): VmcFrame {
    const deltas = this.deltas(world)
    const run = Math.min(Math.max((pace - WALK_PACE) / (RUN_PACE - WALK_PACE), 0), 1)
    const degrees = (walk: number, ran: number): number => (walk + (ran - walk) * run) * this.arms * (Math.PI / 180)
    const opening = degrees(ARMS_AT_WALK.open, ARMS_AT_RUN.open)
    const turning = degrees(ARMS_AT_WALK.twist, ARMS_AT_RUN.twist)
    // Opening rolls each upper arm outward in its shoulder's frame (about the forward axis: +X for the
    // left arm, −X for the right). Twisting turns it outward about its own length, so a bent forearm
    // points ahead instead of across the chest; the forearm twists back about its own length, so the
    // palm keeps facing the thigh. Both T-pose bone axes lie on X, so one pair of twists serves both sides.
    const out = qaxisAngle([1, 0, 0], -turning)
    const back = qinv(out)
    const open = (side: number): Quat => qaxisAngle([0, 0, 1], side * opening)
    const adjust: Record<string, (local: Quat) => Quat> = {
      LeftUpperArm: local => qmul(qmul(open(1), local), out),
      RightUpperArm: local => qmul(qmul(open(-1), local), out),
      LeftLowerArm: local => qmul(local, back),
      RightLowerArm: local => qmul(local, back),
    }
    return {
      hips: scale(toUnityVec(world.pos[this.hips]), this.scale),
      bones: BONES.map((b, i) => {
        const p = BONE_PARENTS[i]
        const local = p < 0 ? deltas[i] : localFromWorld(deltas[p], deltas[i])
        const tweak = adjust[b.bone]
        return [b.bone, toUnityQuat(tweak ? tweak(local) : local)]
      }),
    }
  }
}
