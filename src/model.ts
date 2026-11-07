// The MotionBricks SOMA ONNX model: encodes context and target poses, runs one generation, and
// decodes the result. Ported from NVlabs/motionbricks (Apache-2.0),
// motionbricks/model/soma/kernels.py and motionbricks/model/kernels.py.

import { readFileSync } from 'node:fs'

import * as ort from 'onnxruntime-node'

import {
  add,
  cross,
  ground,
  headingFromHips,
  normalize,
  type Quat,
  qfromBasis,
  qinv,
  qmul,
  qrotate,
  qyaw,
  sub,
  type Vec3,
  wrapAngle,
} from './math.ts'

export const MODEL_FPS = 30
const POSES_PER_TOKEN = 4
const BOUNDARY_FRAMES = POSES_PER_TOKEN - 1
const SLOTS = 2 * POSES_PER_TOKEN
const MIN_TOKENS = 6
const MAX_TOKENS = 16
const FOOT_CONTACTS = 4

/** World positions and rotations of the model's 30 joints, in manifest order (Hips first). */
export interface ModelPose {
  pos: Vec3[]
  rot: Quat[]
}

export interface Generation {
  /** Valid frames at 30 Hz: the first matches the last context pose, the last reaches the first target. */
  frames: Array<{ root: Vec3; rot: Quat[] }>
  /** Per frame: left heel, left toe, right heel, right toe, each 0–1. */
  contacts: number[][]
  tokens: number
  ms: number
}

export class SomaModel {
  readonly joints: readonly string[]
  private readonly session: ort.InferenceSession
  private readonly rightLeg: number
  private readonly leftLeg: number

  private constructor(session: ort.InferenceSession, joints: string[]) {
    this.session = session
    this.joints = joints
    this.rightLeg = joints.indexOf('RightLeg')
    this.leftLeg = joints.indexOf('LeftLeg')
    if (joints[0] !== 'Hips' || this.rightLeg < 0 || this.leftLeg < 0) throw new Error('unexpected SOMA joint layout')
  }

  static async load(onnxPath: string, manifestPath: string): Promise<SomaModel> {
    const manifest: unknown = JSON.parse(readFileSync(manifestPath, 'utf8'))
    const joints =
      typeof manifest === 'object' && manifest !== null && 'joint_names' in manifest ? manifest.joint_names : null
    if (!Array.isArray(joints) || !joints.every(j => typeof j === 'string')) {
      throw new Error(`${manifestPath}: no joint_names`)
    }
    const session = await ort.InferenceSession.create(onnxPath, { executionProviders: ['cpu'] })
    return new SomaModel(session, joints)
  }

  /** Free the ONNX session; call before exiting. */
  release(): Promise<void> {
    return this.session.release()
  }

  /**
   * Generate motion from four context poses (oldest first, 30 Hz apart) toward one to four
   * target poses. `allowedTokens` bounds the length: n tokens give 4n − 6 frames.
   */
  async generate(context: ModelPose[], targets: ModelPose[], allowedTokens: readonly number[]): Promise<Generation> {
    const { feeds, alignP, alignQ } = this.encode(context, targets, allowedTokens)
    const start = performance.now()
    const out = await this.session.run(feeds)
    const ms = performance.now() - start
    const motions = out.pred_global_motions.data
    const tokens = Number(out.pred_num_tokens.data[0])
    if (!(motions instanceof Float32Array)) throw new Error('unexpected output type')
    return { ...this.decode(motions, out.pred_global_motions.dims[2], tokens, alignP, alignQ), tokens, ms }
  }

  /** The model's input tensors for one request, and the alignment to undo on its output. */
  private encode(
    context: ModelPose[],
    targets: ModelPose[],
    allowedTokens: readonly number[]
  ): { feeds: Record<string, ort.Tensor>; alignP: Vec3; alignQ: Quat } {
    if (context.length !== POSES_PER_TOKEN || targets.length < 1 || targets.length > POSES_PER_TOKEN) {
      throw new Error('need 4 context poses and 1–4 targets')
    }
    const slots = [
      ...context,
      ...Array.from({ length: POSES_PER_TOKEN }, (_, i) => targets[Math.min(i, targets.length - 1)]),
    ]
    const J = this.joints.length
    // Per pose: joint positions relative to the hips (3·29), then 6D rotations (6·30).
    const poseWidth = (J - 1) * 3 + J * 6

    // Align so the last context pose stands at the origin facing +X, as the engine does by default.
    const last = slots[POSES_PER_TOKEN - 1]
    const alignP = ground(last.pos[0])
    const alignQ = qyaw(headingFromHips(last.pos[this.rightLeg], last.pos[this.leftLeg]) - Math.PI / 2)
    const toModel = qinv(alignQ)
    const aligned = (p: Vec3): Vec3 => qrotate(toModel, sub(p, alignP))
    const heading = (s: ModelPose): number =>
      headingFromHips(aligned(s.pos[this.rightLeg]), aligned(s.pos[this.leftLeg]))

    const globalRoot = new Float32Array(SLOTS * 5)
    const localRoot = new Float32Array(SLOTS * 4)
    const localPoses = new Float32Array(SLOTS * poseWidth)
    for (let f = 0; f < SLOTS; f++) {
      // Root velocities look one pose ahead, so they are zero at the last context and last target pose.
      const next = f === POSES_PER_TOKEN - 1 ? f : Math.min(f + 1, SLOTS - 1)
      const hips = aligned(slots[f].pos[0])
      const nextHips = aligned(slots[next].pos[0])
      const h = heading(slots[f])
      const turn = wrapAngle(heading(slots[next]) - h)
      globalRoot.set([hips[0], hips[1], hips[2], Math.cos(h), Math.sin(h)], f * 5)
      localRoot.set(
        [turn * MODEL_FPS, (nextHips[0] - hips[0]) * MODEL_FPS, (nextHips[2] - hips[2]) * MODEL_FPS, hips[1]],
        f * 4
      )
      const row = f * poseWidth
      for (let j = 0; j < J; j++) {
        if (j > 0) {
          const p = aligned(slots[f].pos[j])
          localPoses.set([p[0] - hips[0], p[1], p[2] - hips[2]], row + (j - 1) * 3)
        }
        const q = qmul(toModel, slots[f].rot[j])
        localPoses.set([...qrotate(q, [1, 0, 0]), ...qrotate(q, [0, 1, 0])], row + (J - 1) * 3 + j * 6)
      }
    }

    const given = Array.from({ length: SLOTS }, (_, i) => (i < POSES_PER_TOKEN + targets.length ? 1 : 0))
    const localGiven = given.map((v, i) =>
      i === POSES_PER_TOKEN - 1 || i === POSES_PER_TOKEN + targets.length - 1 ? 0 : v
    )
    const allowed = Array.from({ length: MAX_TOKENS - MIN_TOKENS + 1 }, (_, i) =>
      allowedTokens.includes(MIN_TOKENS + i) ? 1 : 0
    )
    const feeds = {
      global_root_values: new ort.Tensor(globalRoot, [1, SLOTS, 5]),
      has_global_root_values: new ort.Tensor(Int32Array.from(given), [1, SLOTS]),
      local_root_values: new ort.Tensor(localRoot, [1, SLOTS, 4]),
      has_local_root_values: new ort.Tensor(Int32Array.from(localGiven), [1, SLOTS]),
      local_poses: new ort.Tensor(localPoses, [1, SLOTS, poseWidth]),
      has_local_poses: new ort.Tensor(Int32Array.from(given), [1, SLOTS]),
      allowed_pred_num_tokens: new ort.Tensor(Int32Array.from(allowed), [1, allowed.length]),
    }
    return { feeds, alignP, alignQ }
  }

  /** Valid frames back in world space: drops the boundary frames on both ends and undoes the alignment. */
  private decode(
    motions: Float32Array,
    width: number,
    tokens: number,
    alignP: Vec3,
    alignQ: Quat
  ): Pick<Generation, 'frames' | 'contacts'> {
    // Per frame: root position + heading (5), joint positions (3·29), 6D rotations (6·30), velocities, contacts (4).
    const J = this.joints.length
    const rotOffset = 5 + (J - 1) * 3
    const count = Math.max(0, tokens * POSES_PER_TOKEN - 2 * BOUNDARY_FRAMES)
    const frames: Generation['frames'] = []
    const contacts: number[][] = []
    for (let i = 0; i < count; i++) {
      const base = (BOUNDARY_FRAMES + i) * width
      const rot: Quat[] = []
      for (let j = 0; j < J; j++) {
        const o = base + rotOffset + j * 6
        const x: Vec3 = [motions[o], motions[o + 1], motions[o + 2]]
        const y: Vec3 = [motions[o + 3], motions[o + 4], motions[o + 5]]
        rot.push(qmul(alignQ, rotationFrom6d(x, y)))
      }
      frames.push({ root: add(qrotate(alignQ, [motions[base], motions[base + 1], motions[base + 2]]), alignP), rot })
      const c = motions.subarray(base + width - FOOT_CONTACTS, base + width)
      contacts.push(Array.from(c, v => Math.min(Math.max(v, 0), 1)))
    }
    return { frames, contacts }
  }
}

function rotationFrom6d(xRaw: Vec3, yRaw: Vec3): Quat {
  const x = normalize(xRaw)
  const z = normalize(cross(x, yRaw))
  return qfromBasis(x, cross(z, x), z)
}
