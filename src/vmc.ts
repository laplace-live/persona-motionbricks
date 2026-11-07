// A minimal VMC (OSC over UDP) sender: one bundle per frame with the root, every bone, and the
// OK/T heartbeat other VMC receivers expect. Persona reads /VMC/Ext/Bone/Pos and /VMC/Ext/Root/Pos.

import { createSocket, type Socket } from 'node:dgram'

import type { VmcFrame } from './retarget.ts'

type Arg = string | number

function oscString(s: string): Buffer {
  const bytes = Buffer.from(s, 'ascii')
  const out = Buffer.alloc((bytes.length + 4) & ~3)
  bytes.copy(out)
  return out
}

/** Numbers are sent as float32 except where the type tag says `i`. */
export function oscMessage(address: string, tags: string, args: Arg[]): Buffer {
  const parts = [oscString(address), oscString(`,${tags}`)]
  args.forEach((arg, i) => {
    if (tags[i] === 's') parts.push(oscString(String(arg)))
    else {
      const b = Buffer.alloc(4)
      if (tags[i] === 'i') b.writeInt32BE(Number(arg))
      else b.writeFloatBE(Number(arg))
      parts.push(b)
    }
  })
  return Buffer.concat(parts)
}

export function oscBundle(messages: Buffer[]): Buffer {
  const timetag = Buffer.alloc(8)
  timetag.writeUInt32BE(1, 4) // "immediately"
  const parts = [oscString('#bundle'), timetag]
  for (const m of messages) {
    const size = Buffer.alloc(4)
    size.writeInt32BE(m.length)
    parts.push(size, m)
  }
  return Buffer.concat(parts)
}

export function vmcPacket(frame: VmcFrame, time: number): Buffer {
  return oscBundle([
    oscMessage('/VMC/Ext/OK', 'i', [1]),
    oscMessage('/VMC/Ext/T', 'f', [time]),
    // The root stays at the origin, unrotated; the Hips bone carries the position.
    oscMessage('/VMC/Ext/Root/Pos', 'sfffffff', ['root', 0, 0, 0, 0, 0, 0, 1]),
    ...frame.bones.map(([bone, q]) =>
      oscMessage('/VMC/Ext/Bone/Pos', 'sfffffff', [bone, ...(bone === 'Hips' ? frame.hips : [0, 0, 0]), ...q])
    ),
  ])
}

export class VmcSender {
  private readonly socket: Socket = createSocket('udp4')
  private readonly host: string
  private readonly port: number
  private readonly start = performance.now()

  constructor(host: string, port: number) {
    this.host = host
    this.port = port
  }

  send(frame: VmcFrame): void {
    this.socket.send(vmcPacket(frame, (performance.now() - this.start) / 1000), this.port, this.host)
  }

  close(): void {
    this.socket.close()
  }
}
