import assert from 'node:assert/strict'
import { test } from 'bun:test'

import type { VmcFrame } from '../src/retarget.ts'
import { vmcPacket } from '../src/vmc.ts'

type Message = { address: string; args: Array<string | number> }

/** Just enough OSC to read back one bundle of string/float/int messages. */
function decodeBundle(buf: Buffer): Message[] {
  let at = 0
  const readString = (): string => {
    const end = buf.indexOf(0, at)
    const s = buf.toString('ascii', at, end)
    at = (end + 4) & ~3
    return s
  }
  assert.equal(readString(), '#bundle')
  at += 8 // timetag
  const messages: Message[] = []
  while (at < buf.length) {
    const end = at + 4 + buf.readInt32BE(at)
    at += 4
    const address = readString()
    const args: Message['args'] = []
    for (const tag of readString().slice(1)) {
      if (tag === 's') args.push(readString())
      else {
        args.push(tag === 'i' ? buf.readInt32BE(at) : buf.readFloatBE(at))
        at += 4
      }
    }
    messages.push({ address, args })
    at = end
  }
  return messages
}

function assertClose(got: Message['args'], want: number[]): void {
  for (const [i, v] of want.entries()) {
    assert.ok(Math.abs(Number(got[i]) - v) < 1e-6, `${String(got[i])} ≠ ${v}`)
  }
}

test('a VMC packet carries the heartbeat, root, and every bone', () => {
  const frame: VmcFrame = {
    hips: [0.25, 0.9, -1.5],
    bones: [
      ['Hips', [0, 0.6, 0, 0.8]],
      ['LeftUpperArm', [0.1, 0.2, 0.3, 0.9274]],
    ],
  }
  const messages = decodeBundle(vmcPacket(frame, 1.5))
  assert.deepEqual(
    messages.map(m => m.address),
    ['/VMC/Ext/OK', '/VMC/Ext/T', '/VMC/Ext/Root/Pos', '/VMC/Ext/Bone/Pos', '/VMC/Ext/Bone/Pos']
  )
  assert.deepEqual(messages[0].args, [1])
  const [hips, arm] = messages.slice(3)
  assert.equal(hips.args[0], 'Hips')
  assertClose(hips.args.slice(1), [0.25, 0.9, -1.5, 0, 0.6, 0, 0.8])
  assert.equal(arm.args[0], 'LeftUpperArm')
  assertClose(arm.args.slice(1), [0, 0, 0, 0.1, 0.2, 0.3, 0.9274])
})
