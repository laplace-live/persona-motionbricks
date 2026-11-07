// The SOMA skeleton as the BVH clips define it: Y up, metres, rooted at Hips.

import { readFileSync } from 'node:fs'

import {
  add,
  headingFromHips,
  identity,
  lerp3,
  type Quat,
  qaxisAngle,
  qinv,
  qmul,
  qrotate,
  qslerp,
  scale,
  type Vec3,
} from './math.ts'

export interface Skeleton {
  /** Joint names, parents before children; Hips is joint 0. */
  names: string[]
  /** Parent index per joint; -1 for Hips. */
  parents: number[]
  /** Rest offset from the parent, in the parent's frame (metres). */
  offsets: Vec3[]
  index: Map<string, number>
}

/** One frame: the root's world position and every joint's local rotation (Hips: world). */
export interface Pose {
  root: Vec3
  local: Quat[]
}

/** World-space position and rotation per joint. */
export interface WorldPose {
  pos: Vec3[]
  rot: Quat[]
}

export interface Clip {
  fps: number
  frames: Pose[]
}

interface RawJoint {
  name: string | null
  parent: number
  offset: Vec3
  channels: string[]
}

const AXES: Vec3[] = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
]

/** Read a BVH file, dropping any joints above Hips (the SOMA clips carry a `Root`) and end sites. */
export function loadBvh(path: string): { skeleton: Skeleton; clip: Clip } {
  const lines = readFileSync(path, 'utf8').split(/\r?\n/)
  const raw: RawJoint[] = []
  const stack: number[] = []
  let pending = -1
  let at = 0
  for (; at < lines.length; at++) {
    const tok = lines[at].trim().split(/\s+/)
    if (tok[0] === 'ROOT' || tok[0] === 'JOINT' || tok[0] === 'End') {
      raw.push({ name: tok[0] === 'End' ? null : tok[1], parent: stack.at(-1) ?? -1, offset: [0, 0, 0], channels: [] })
      pending = raw.length - 1
    } else if (tok[0] === '{') stack.push(pending)
    else if (tok[0] === '}') stack.pop()
    else if (tok[0] === 'OFFSET')
      raw[stack.at(-1) ?? 0].offset = scale([Number(tok[1]), Number(tok[2]), Number(tok[3])], 0.01)
    else if (tok[0] === 'CHANNELS') raw[stack.at(-1) ?? 0].channels = tok.slice(2)
    else if (tok[0] === 'MOTION') break
  }
  const frameCount = Number(lines[at + 1].split(/\s+/).at(-1))
  const frameTime = Number(lines[at + 2].split(/\s+/).at(-1))

  const hipsRaw = raw.findIndex(j => j.name === 'Hips')
  if (hipsRaw < 0) throw new Error(`${path}: no Hips joint`)
  // Keep Hips and its named descendants; raw order already lists parents first.
  const keep = new Map<number, number>([[hipsRaw, 0]])
  const skeleton: Skeleton = { names: ['Hips'], parents: [-1], offsets: [[0, 0, 0]], index: new Map([['Hips', 0]]) }
  raw.forEach((j, i) => {
    if (i <= hipsRaw || j.name === null || !keep.has(j.parent)) return
    keep.set(i, skeleton.names.length)
    skeleton.index.set(j.name, skeleton.names.length)
    skeleton.names.push(j.name)
    skeleton.parents.push(keep.get(j.parent) ?? -1)
    skeleton.offsets.push(j.offset)
  })

  const frames: Pose[] = []
  const worldRot: Quat[] = []
  const worldPos: Vec3[] = []
  for (let f = 0; f < frameCount; f++) {
    const values = lines[at + 3 + f].trim().split(/\s+/).map(Number)
    let col = 0
    const local: Quat[] = new Array(skeleton.names.length)
    raw.forEach((j, i) => {
      let rot = identity()
      let pos: Vec3 | null = null
      for (const ch of j.channels) {
        const v = values[col++]
        const axis = 'XYZ'.indexOf(ch[0])
        if (ch.endsWith('position')) {
          pos ??= [0, 0, 0]
          pos[axis] = v * 0.01
        } else rot = qmul(rot, qaxisAngle(AXES[axis], (v * Math.PI) / 180))
      }
      // Only Hips needs a world transform, to fold in the joints above it; parents come first.
      if (i <= hipsRaw) {
        const offset = pos ?? j.offset
        worldRot[i] = j.parent < 0 ? rot : qmul(worldRot[j.parent], rot)
        worldPos[i] = j.parent < 0 ? offset : add(worldPos[j.parent], qrotate(worldRot[j.parent], offset))
      }
      const k = keep.get(i)
      if (k !== undefined) local[k] = rot
    })
    local[0] = worldRot[hipsRaw]
    frames.push({ root: worldPos[hipsRaw], local })
  }
  return { skeleton, clip: { fps: 1 / frameTime, frames } }
}

export function forward(skeleton: Skeleton, pose: Pose): WorldPose {
  const pos: Vec3[] = new Array(skeleton.names.length)
  const rot: Quat[] = new Array(skeleton.names.length)
  for (let j = 0; j < skeleton.names.length; j++) {
    const p = skeleton.parents[j]
    if (p < 0) {
      pos[j] = pose.root
      rot[j] = pose.local[j]
    } else {
      rot[j] = qmul(rot[p], pose.local[j])
      pos[j] = add(pos[p], qrotate(rot[p], skeleton.offsets[j]))
    }
  }
  return { pos, rot }
}

/** A joint's index by name; throws when the skeleton lacks it. */
export function jointIndex(skeleton: Skeleton, name: string): number {
  const j = skeleton.index.get(name)
  if (j === undefined) throw new Error(`skeleton lacks ${name}`)
  return j
}

/** A world pose's heading, from its hip joints (see headingFromHips). */
export const poseHeading = (skeleton: Skeleton, world: WorldPose): number =>
  headingFromHips(world.pos[jointIndex(skeleton, 'RightLeg')], world.pos[jointIndex(skeleton, 'LeftLeg')])

export const blendPose = (a: Pose, b: Pose, t: number): Pose => ({
  root: lerp3(a.root, b.root, t),
  local: a.local.map((q, j) => qslerp(q, b.local[j], t)),
})

/** The pose at `time` seconds, interpolated and clamped to the clip. */
export function samplePose(frames: Pose[], fps: number, time: number): Pose {
  const f = Math.min(Math.max(time * fps, 0), frames.length - 1)
  const i = Math.floor(f)
  return i >= frames.length - 1 ? frames[frames.length - 1] : blendPose(frames[i], frames[i + 1], f - i)
}

/** Apply a world rigid transform (rotation about the origin, then translation) to a pose's root. */
export const moveRoot = (pose: Pose, rotation: Quat, translation: Vec3): Pose => ({
  root: add(qrotate(rotation, pose.root), translation),
  local: [qmul(rotation, pose.local[0]), ...pose.local.slice(1)],
})

/** Local rotations from world rotations, for joints whose parent's world rotation is known. */
export const localFromWorld = (parentWorld: Quat, world: Quat): Quat => qmul(qinv(parentWorld), world)
