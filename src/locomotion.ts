// A receding-horizon locomotion loop after NVIDIA's docs/examples/locomotion_loop.py: each
// generation steers toward target poses placed a second of travel ahead, and the next one takes
// over a few frames before the current motion ends, or as soon as the command changes.
// A jump is a smart-primitive sequence instead: target windows reached one after another, after
// which locomotion resumes.

import {
  add,
  dot,
  ground,
  headingDir,
  headingFromDir,
  identity,
  length,
  type Quat,
  qinv,
  qmul,
  qnormalize,
  qrotate,
  qslerp,
  qyaw,
  scale,
  sub,
  type Vec3,
} from './math.ts'
import { type Generation, MODEL_FPS, type ModelPose, type SomaModel } from './model.ts'
import {
  type Clip,
  forward,
  jointIndex,
  localFromWorld,
  moveRoot,
  type Pose,
  poseHeading,
  type Skeleton,
  samplePose,
} from './skeleton.ts'

export const UPDATE_HZ = 60
/** Shown poses per model frame: the context takes every other pose shown. */
const CONTEXT_STRIDE = UPDATE_HZ / MODEL_FPS
const LEAD_SECONDS = 1
const TRANSITION_EARLY_FRAMES = 8
const TURN_THRESHOLD = (15 * Math.PI) / 180
const SPEED_THRESHOLD = 0.3
export const ALLOWED_TOKENS = [8, 9, 10, 11, 12, 13, 14]
/**
 * Generated motion sometimes pushes a joint below the floor, landing a running jump by about 4 cm.
 * No joint centre may go lower than this (m); the whole pose rises to meet it.
 */
const FLOOR = 0
/** Time constant (s) of the smoothed ground speed, long enough to ride out each stride's surge. */
const PACE_SMOOTHING = 0.5
export type Gear = 'idle' | 'walk' | 'jog' | 'run'

/** What the character should do: travel along `dir` (unit, on the ground) at `speed` m/s. */
export interface Command {
  gear: Gear
  speed: number
  dir: Vec3 | null
  /** Jump now, along `dir` or, standing, straight ahead; ignored while a jump is under way. */
  jump?: boolean
}

/** Four target poses 30 Hz apart; canonical windows start at the origin travelling toward +Z. */
export interface TargetWindow {
  poses: Pose[]
  /** Mean ground speed over the window, m/s. */
  speed: number
}

/** One step of a move: a target window in the move's canonical frame, and how to reach it. */
export interface MoveStep {
  window: TargetWindow
  /** Frames before the step's end at which the next one takes over. */
  handover: number
  /** Token counts the generation may use: n tokens give 4n − 6 frames at 30 Hz. */
  tokens: number[]
}

/** A smart-primitive sequence, placed with its origin `lead` metres ahead of the character, facing its way. */
export interface Move {
  steps: MoveStep[]
  /** Ground distance (m) to the move's origin, given the character's pace (m/s). */
  lead: (pace: number) => number
}

/** Maps skeleton poses onto the model's joints and back. */
export class Rig {
  readonly skeleton: Skeleton
  readonly modelJoints: number[]
  /** Model joint driving each skeleton joint; -1 keeps the reference rotation (fingers). */
  private readonly modelOf: number[]
  private readonly reference: Pose
  /** Joints that can touch the floor: everything but the finger chains. */
  private readonly body: number[]

  constructor(skeleton: Skeleton, joints: readonly string[], reference: Pose) {
    this.skeleton = skeleton
    this.reference = reference
    this.body = skeleton.names.flatMap((name, j) => (/Hand(Thumb|Index|Middle|Ring|Pinky)/.test(name) ? [] : [j]))
    this.modelJoints = joints.map(name => jointIndex(skeleton, name))
    this.modelOf = skeleton.names.map(() => -1)
    // As the engine binds it: a model joint is driven only when its parent is, so the
    // finger tips the model tracks for hand orientation keep the reference chain.
    this.modelJoints.forEach((j, m) => {
      const p = skeleton.parents[j]
      if (p < 0 || this.modelOf[p] >= 0) this.modelOf[j] = m
    })
  }

  /** The world positions and rotations of a pose's model joints. */
  modelPose(pose: Pose): ModelPose {
    const world = forward(this.skeleton, pose)
    return { pos: this.modelJoints.map(j => world.pos[j]), rot: this.modelJoints.map(j => world.rot[j]) }
  }

  fromGenerated(frame: Generation['frames'][number]): Pose {
    const local = this.skeleton.names.map((_, j): Quat => {
      const m = this.modelOf[j]
      if (m < 0) return this.reference.local[j]
      if (j === 0) return frame.rot[0]
      return localFromWorld(frame.rot[this.modelOf[this.skeleton.parents[j]]], frame.rot[m])
    })
    return { root: frame.root, local }
  }

  /** The pose raised so its lowest joint (finger tips aside) is no lower than `FLOOR`. */
  onFloor(pose: Pose): Pose {
    const world = forward(this.skeleton, pose)
    let lowest = Infinity
    for (const j of this.body) lowest = Math.min(lowest, world.pos[j][1])
    return lowest >= FLOOR
      ? pose
      : { root: [pose.root[0], pose.root[1] + FLOOR - lowest, pose.root[2]], local: pose.local }
  }

  /** Ground position and heading of a pose. */
  where(pose: Pose): { at: Vec3; heading: number } {
    return { at: ground(pose.root), heading: poseHeading(this.skeleton, forward(this.skeleton, pose)) }
  }
}

/**
 * Four-pose windows from `clip` at each of `times`, moved together so the first starts at the
 * origin travelling toward +Z (a still first window keeps its hips' facing instead).
 */
export function canonicalWindows(rig: Rig, clip: Clip, times: number[]): TargetWindow[] {
  const windows = times.map(t => [0, 1, 2, 3].map(k => samplePose(clip.frames, clip.fps, t + k / MODEL_FPS)))
  const speedOf = (poses: Pose[]): number => (length(sub(ground(poses[3].root), ground(poses[0].root))) * MODEL_FPS) / 3
  const first = windows[0]
  const start = ground(first[0].root)
  const travel = sub(ground(first[3].root), start)
  const heading = speedOf(first) > 0.2 ? headingFromDir(travel) : rig.where(first[0]).heading
  const turn = qyaw(-heading)
  const shift = scale(qrotate(turn, start), -1)
  return windows.map(poses => ({ poses: poses.map(p => moveRoot(p, turn, shift)), speed: speedOf(poses) }))
}

/** Scale a window's ground travel to `speed` m/s, as motionbricks.runtime.target_pose does. */
function adjustSpeed(window: TargetWindow, speed: number): Pose[] {
  const k = window.speed > 1e-6 ? speed / window.speed : 1
  const [x0, , z0] = window.poses[0].root
  return window.poses.map(p => ({
    root: [x0 + (p.root[0] - x0) * k, p.root[1], z0 + (p.root[2] - z0) * k],
    local: p.local,
  }))
}

/** Blend the generated motion so it starts on the shown pose and ends on the first target. */
function correctEndpoints(frames: Pose[], shown: Pose, target: Pose): Pose[] {
  const n = frames.length
  if (n < 2) return frames
  const first = frames[0]
  const last = frames[n - 1]
  const startRot = shown.local.map((q, j) => qmul(qinv(first.local[j]), q))
  const endRot = target.local.map((q, j) => qmul(qinv(last.local[j]), q))
  const startPos = sub(shown.root, first.root)
  const endPos = sub(target.root, last.root)
  return frames.map((f, i) => {
    const end = i / (n - 1)
    const start = 1 - end
    return {
      root: add(add(f.root, scale(startPos, start)), scale(endPos, end)),
      local: f.local.map((q, j) =>
        qnormalize(qmul(qmul(q, qslerp(identity(), startRot[j], start)), qslerp(identity(), endRot[j], end)))
      ),
    }
  })
}

export class Locomotion {
  current: Pose
  /** Ground speed of the shown motion, m/s, smoothed over about half a second. */
  pace = 0
  lastGeneration: Generation | null = null
  generations = 0
  private readonly model: SomaModel
  private readonly rig: Rig
  private readonly windows: Record<Gear, TargetWindow[]>
  /** Picks the next jump variation. */
  private readonly jump: () => Move
  /** The last poses shown at UPDATE_HZ, oldest first: just enough for four context poses 30 Hz apart. */
  private readonly history: Pose[]
  private segment: Pose[]
  private segmentTime = 0
  /** Frames before the current segment's end at which the next generation takes over. */
  private handover = TRANSITION_EARLY_FRAMES
  private last: Command | null = null
  private windowIndex = 0
  /** The jump in progress: its steps still to generate, with their targets placed in the world. */
  private move: Array<{ targets: Pose[]; tokens: number[]; handover: number }> | null = null

  constructor(model: SomaModel, rig: Rig, windows: Record<Gear, TargetWindow[]>, jump: () => Move, start: Pose) {
    this.model = model
    this.rig = rig
    this.windows = windows
    this.jump = jump
    this.current = start
    this.segment = [start]
    this.history = Array.from({ length: 3 * CONTEXT_STRIDE + 1 }, () => start)
  }

  /** A jump is under way; commands wait until it lands. */
  get jumping(): boolean {
    return this.move !== null
  }

  /** Advance one 1/UPDATE_HZ step, generating first when the command, a move, or the clock asks for it. */
  async step(command: Command): Promise<void> {
    const dt = 1 / UPDATE_HZ
    const due = this.timeToTransition() <= dt
    if (command.jump && !this.move) {
      this.move = this.placeMove(this.jump(), command)
      await this.nextMoveStep(command)
    } else if (this.move) {
      if (due) await this.nextMoveStep(command)
    } else if (due || this.commandChanged(command)) await this.generate(command)
    this.segmentTime += dt
    const previous = this.current
    this.current = this.rig.onFloor(samplePose(this.segment, MODEL_FPS, this.segmentTime))
    const speed = length(sub(ground(this.current.root), ground(previous.root))) / dt
    this.pace += (speed - this.pace) * (1 - Math.exp(-dt / PACE_SMOOTHING))
    this.history.push(this.current)
    this.history.shift()
  }

  where(): { at: Vec3; heading: number } {
    return this.rig.where(this.current)
  }

  private timeToTransition(): number {
    return Math.max(0, this.segment.length - 1 - this.handover) / MODEL_FPS - this.segmentTime
  }

  private commandChanged(command: Command): boolean {
    const last = this.last
    if (!last || last.gear !== command.gear || Math.abs(last.speed - command.speed) > SPEED_THRESHOLD) return true
    return !!command.dir && !!last.dir && Math.acos(Math.min(1, dot(command.dir, last.dir))) > TURN_THRESHOLD
  }

  /** The move's steps, with their targets placed along the command, or straight ahead from a stand. */
  private placeMove(move: Move, command: Command): NonNullable<Locomotion['move']> {
    const { at, heading: facing } = this.where()
    const dir = command.gear !== 'idle' && command.dir ? command.dir : headingDir(facing)
    const turn = qyaw(headingFromDir(dir))
    const origin = add(at, scale(dir, move.lead(this.pace)))
    return move.steps.map(({ window, tokens, handover }) => ({
      targets: window.poses.map(p => moveRoot(p, turn, origin)),
      tokens,
      handover,
    }))
  }

  /** Generate toward the jump's next step or, once the last one has played out, resume locomotion. */
  private async nextMoveStep(command: Command): Promise<void> {
    const step = this.move?.shift()
    if (step) await this.generateToward(step.targets, step.tokens, step.handover)
    else {
      this.move = null
      await this.generate(command)
    }
  }

  private async generate(command: Command): Promise<void> {
    const { at, heading: facing } = this.where()
    const dir = command.gear === 'idle' || command.speed <= 0 ? null : command.dir
    const options = this.windows[dir ? command.gear : 'idle']
    const window = options[this.windowIndex++ % options.length]
    const heading = dir ? headingFromDir(dir) : facing
    const destination = dir ? add(at, scale(dir, command.speed * LEAD_SECONDS)) : at
    const targets = adjustSpeed(window, dir ? command.speed : 0).map(p => moveRoot(p, qyaw(heading), destination))
    await this.generateToward(targets, ALLOWED_TOKENS, TRANSITION_EARLY_FRAMES)
    this.last = { ...command }
  }

  /** Generate from the last four shown poses toward `targets`, and play the result from its start. */
  private async generateToward(targets: Pose[], tokens: number[], handover: number): Promise<void> {
    const context = [0, 1, 2, 3].map(k => this.history[k * CONTEXT_STRIDE])
    const asModel = (p: Pose): ModelPose => this.rig.modelPose(p)
    const generation = await this.model.generate(context.map(asModel), targets.map(asModel), tokens)
    this.segment = correctEndpoints(
      generation.frames.map(f => this.rig.fromGenerated(f)),
      context[3],
      targets[0]
    )
    this.segmentTime = 0
    this.handover = handover
    this.lastGeneration = generation
    this.generations++
  }
}
