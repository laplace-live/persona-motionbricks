// What the character is asked to do: terminal keys, or a scripted tour for hands-free demos.
// Directions are world axes as a front-facing avatar sees them: +Z walks toward the camera.

import type { Command, Gear } from './locomotion.ts'
import { length, normalize, scale, type Vec3 } from './math.ts'

type Moving = Exclude<Gear, 'idle'>

export const GEAR_SPEED: Record<Moving, number> = { walk: 1.3, jog: 2, run: 3.5 }

const AWAY: Vec3 = [0, 0, -1]
const TOWARD: Vec3 = [0, 0, 1]
/** The viewer's left is the avatar's right (−X) while it faces the camera. */
const SCREEN_LEFT: Vec3 = [-1, 0, 0]
const SCREEN_RIGHT: Vec3 = [1, 0, 0]
const HOME_RADIUS = 0.3

const idle: Command = { gear: 'idle', speed: 0, dir: null }

/** Keys as the terminal sends them, lowercased; arrows arrive as escape sequences. */
const KEY_DIRS = new Map<string, Vec3>([
  ['w', AWAY],
  ['\u001b[a', AWAY],
  ['s', TOWARD],
  ['\u001b[b', TOWARD],
  ['a', SCREEN_LEFT],
  ['\u001b[d', SCREEN_LEFT],
  ['d', SCREEN_RIGHT],
  ['\u001b[c', SCREEN_RIGHT],
])
const KEY_GEARS = new Map<string, Moving>([
  ['1', 'walk'],
  ['2', 'jog'],
  ['3', 'run'],
])

/** Walk back to where the stream started; idle once there. */
function homeward(at: Vec3): Command {
  return length(at) < HOME_RADIUS ? idle : { gear: 'walk', speed: GEAR_SPEED.walk, dir: normalize(scale(at, -1)) }
}

export interface Controller {
  /** The command for a character standing at `at` (ground, metres), `time` seconds into the run. */
  command(at: Vec3, time: number): Command
  readonly label: string
  /** Release whatever the controller holds, such as the terminal. */
  close?(): void
}

/** Arrow keys or WASD set a direction, 1/2/3 a gear, J jumps, space stops, H walks home, Q quits. */
export class KeyboardController implements Controller {
  label = 'idle'
  private gear: Moving = 'walk'
  private dir: Vec3 | null = null
  private home = false
  private jumpPressed = false
  private readonly onQuit: () => void

  constructor(onQuit: () => void) {
    if (!process.stdin.isTTY) throw new Error('keyboard control needs a terminal; use --tour')
    this.onQuit = onQuit
    process.stdin.setRawMode(true)
    process.stdin.resume()
    process.stdin.on('data', this.onKey)
  }

  private readonly onKey = (data: Buffer): void => {
    const key = data.toString().toLowerCase()
    const dir = KEY_DIRS.get(key)
    const gear = KEY_GEARS.get(key)
    if (key === 'q' || key === '\u0003') this.onQuit()
    else if (dir) {
      this.dir = dir
      this.home = false
    } else if (gear) this.gear = gear
    else if (key === 'j') this.jumpPressed = true
    else if (key === ' ') {
      this.dir = null
      this.home = false
    } else if (key === 'h') this.home = true
  }

  command(at: Vec3): Command {
    const jump = this.jumpPressed
    this.jumpPressed = false
    if (this.home) {
      const next = homeward(at)
      this.home = next.dir !== null
      this.dir = null
      this.label = this.home ? 'walking home' : 'idle'
      return { ...next, jump }
    }
    this.label = this.dir ? this.gear : 'idle'
    return this.dir ? { gear: this.gear, speed: GEAR_SPEED[this.gear], dir: this.dir, jump } : { ...idle, jump }
  }

  close(): void {
    process.stdin.off('data', this.onKey)
    process.stdin.setRawMode(false)
    process.stdin.pause()
  }
}

type TourStep = [seconds: number, gear: Moving | 'idle' | 'home', dir: Vec3 | null, jump?: 'jump']

/** A loop for hands-free demos: a walked square with a jump, a jog with a jump, then home and rest. */
export class TourController implements Controller {
  label = ''
  private readonly steps: TourStep[] = [
    [2, 'idle', null],
    [1.5, 'walk', SCREEN_RIGHT],
    [1.5, 'walk', AWAY],
    [2, 'walk', SCREEN_LEFT, 'jump'],
    [1.5, 'walk', TOWARD],
    [1.5, 'idle', null],
    [2, 'jog', SCREEN_RIGHT, 'jump'],
    [3.5, 'home', null],
    [1.5, 'idle', null],
  ]
  private readonly total = this.steps.reduce((t, [s]) => t + s, 0)
  /** Loop number and step index last seen, so each jump fires once on entering its step. */
  private seen = ''

  command(at: Vec3, time: number): Command {
    let t = time % this.total
    for (const [i, [seconds, gear, dir, jumps]] of this.steps.entries()) {
      if (t < seconds) {
        const step = `${Math.floor(time / this.total)}:${i}`
        const jump = jumps === 'jump' && step !== this.seen
        this.seen = step
        this.label = `tour: ${gear}`
        if (gear === 'home') return homeward(at)
        return gear === 'idle' || !dir ? { ...idle, jump } : { gear, speed: GEAR_SPEED[gear], dir, jump }
      }
      t -= seconds
    }
    return idle
  }
}
