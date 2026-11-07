// MotionBricks → VMC: generates locomotion with NVIDIA's model on the CPU and streams it to
// LAPLACE Persona (or any VMC receiver) at 60 Hz.

import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { parseArgs } from 'node:util'
import { personaWsUrl } from '@laplace.live/persona-sdk'
import { ASSETS_DIR, modelFile } from './assets.ts'
import { type Controller, KeyboardController, TourController } from './controls.ts'
import { createDemo } from './demo.ts'
import { UPDATE_HZ } from './locomotion.ts'
import { linkPersona, type PersonaLink } from './persona.ts'
import { forward } from './skeleton.ts'
import { VmcSender } from './vmc.ts'

const KEYS = 'Keys: arrows/WASD walk that way · 1 walk 2 jog 3 run · j jump · space stop · h home · q quit'
const HELP = `Usage: bun start [options]

  --tour             hands-free loop instead of keyboard control
  --host <ip>        VMC receiver host (default 127.0.0.1)
  --port <n>         VMC receiver port (default 39539)
  --persona          set Persona up over the plugin API (API key in PERSONA_TOKEN)
  --url <ws-url>     plugin API address (default ${personaWsUrl()})
  --instance <id>    VRM instance to drive (default: the first VRM in the scene)
  --scale <k>        avatar hips height ÷ 0.99 m, so travel matches its legs
                     (default 0.9, typical for VRoid avatars)
  --arms <k>         scale how far arms open to clear the chest and hips (default 1, 0 = off)
  --fp32             use the FP32 model (fetch it with: bun run fetch-assets --fp32)
  --duration <s>     stop after this many seconds
  --assets <dir>     where fetch-assets put the model and clips (default ./assets)

${KEYS}`

const { values: args } = parseArgs({
  options: {
    tour: { type: 'boolean', default: false },
    host: { type: 'string', default: '127.0.0.1' },
    port: { type: 'string', default: '39539' },
    persona: { type: 'boolean', default: false },
    url: { type: 'string', default: personaWsUrl() },
    instance: { type: 'string' },
    scale: { type: 'string', default: '0.9' },
    arms: { type: 'string', default: '1' },
    fp32: { type: 'boolean', default: false },
    duration: { type: 'string' },
    assets: { type: 'string', default: ASSETS_DIR },
    help: { type: 'boolean', short: 'h', default: false },
  },
})
if (args.help) {
  console.log(HELP)
  process.exit(0)
}
if (!existsSync(join(args.assets, modelFile(args.fp32)))) {
  console.error(`No MotionBricks model in ${args.assets}. Run: bun run fetch-assets${args.fp32 ? ' --fp32' : ''}`)
  process.exit(1)
}

console.log('Loading MotionBricks…')
const { model, locomotion, retargeter, skeleton } = await createDemo(args.assets, {
  fp32: args.fp32,
  arms: Number(args.arms),
  scale: Number(args.scale),
})

let link: PersonaLink | null = null
let port = Number(args.port)
const token = process.env.PERSONA_TOKEN
if (args.persona || token) {
  if (!token) {
    console.error('--persona needs a plugin API key in PERSONA_TOKEN (Persona: Advanced → Plugin API).')
    process.exit(1)
  }
  try {
    link = await linkPersona({ token, url: args.url, instanceId: args.instance, log: m => console.warn(m) })
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error)
    console.error(`Couldn't set Persona up over the plugin API at ${args.url}: ${reason}.
Is Persona running with Advanced → Plugin API on, and is the key current?`)
    process.exit(1)
  }
  port = link.port
  console.log(`Persona: a new VMC source on port ${port} drives VRM ${link.instanceId}; it is removed on exit.`)
} else {
  console.log(`Persona: in Pose Tracking, add a VMC source on port ${port}, assign it to your VRM avatar,
and turn on Enable Pose Tracking. (Or set PERSONA_TOKEN to let this demo do that over the plugin API.)`)
}

const sender = new VmcSender(args.host, port)
let running = true
const stop = (): void => {
  running = false
}
const controller: Controller = args.tour ? new TourController() : new KeyboardController(stop)
process.on('SIGINT', stop)
console.log(
  `Streaming VMC to ${args.host}:${port} at ${UPDATE_HZ} Hz. ${args.tour ? 'Touring; Ctrl-C to stop.' : KEYS}\n`
)

const stepMs = 1000 / UPDATE_HZ
const endAt = args.duration ? Number(args.duration) : Infinity
let time = 0
let next = performance.now()
let shownAt = 0
while (running && time < endAt) {
  const now = performance.now()
  if (now - next > 250) next = now // after a stall, resume rather than fast-forward
  let stepped = false
  while (next <= now && running) {
    await locomotion.step(controller.command(locomotion.where().at, time))
    time += 1 / UPDATE_HZ
    next += stepMs
    stepped = true
  }
  if (stepped) sender.send(retargeter.frame(forward(skeleton, locomotion.current), locomotion.pace))
  if (now - shownAt > 250) {
    shownAt = now
    const { at, heading } = locomotion.where()
    const gen = locomotion.lastGeneration
    const facing = ((heading * 180) / Math.PI).toFixed(0).padStart(4)
    const where = `at (${at[0].toFixed(2)}, ${at[2].toFixed(2)}) m, facing ${facing}°`
    const stats = gen
      ? ` · ${locomotion.generations} generations, last ${gen.ms.toFixed(0)} ms → ${gen.frames.length} frames`
      : ''
    const label = locomotion.jumping ? 'jumping' : controller.label
    process.stdout.write(`\r\u001b[K${label.padEnd(13)} ${where}${stats}`)
  }
  await Bun.sleep(Math.max(1, next - performance.now()))
}

process.stdout.write('\n')
controller.close?.()
sender.close()
await link?.close()
await model.release()
process.exit(0)
