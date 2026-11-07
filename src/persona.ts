// Optional plugin-API setup: create a VMC body source in Persona, bind it to a VRM avatar and
// switch body tracking on, then put everything back on exit.

import { createNodeClient } from '@laplace.live/persona-sdk'

export interface PersonaLink {
  port: number
  instanceId: string
  close(): Promise<void>
}

export async function linkPersona(options: {
  token: string
  url: string
  instanceId?: string
  log: (message: string) => void
}): Promise<PersonaLink> {
  const client = createNodeClient({
    token: options.token,
    url: options.url,
    log: options.log,
    clientInfo: { name: 'MotionBricks VMC demo', version: '0.1.0' },
  })
  try {
    await client.connect()
    const { instances } = await client.call('instance.list')
    const vrms = instances.filter(i => i.kind === 'model' && i.format === 'vrm')
    const target = options.instanceId ? vrms.find(i => i.instanceId === options.instanceId) : vrms[0]
    if (!target)
      throw new Error(
        options.instanceId ? `no VRM instance ${options.instanceId}` : 'no VRM avatar in the active scene'
      )
    const { item } = await client.call('instance.get', { instanceId: target.instanceId })
    const previousSource = 'poseSourceId' in item ? item.poseSourceId : null
    const { settings } = await client.call('settings.get')
    const poseWasEnabled = settings.pose.enabled

    const { source } = await client.call('tracking.addSource', { kind: 'vmc', name: 'MotionBricks' })
    if (source.port === null) throw new Error('Persona gave the VMC source no port')
    await client.call('instance.setTrackingSources', { instanceId: target.instanceId, poseSourceId: source.id })
    if (!poseWasEnabled) await client.call('pose.setEnabled', { enabled: true })

    return {
      port: source.port,
      instanceId: target.instanceId,
      async close() {
        try {
          await client.call('instance.setTrackingSources', {
            instanceId: target.instanceId,
            poseSourceId: previousSource,
          })
          await client.call('tracking.removeSource', { id: source.id })
          if (!poseWasEnabled) await client.call('pose.setEnabled', { enabled: false })
        } finally {
          client.close()
        }
      },
    }
  } catch (error) {
    client.close()
    throw error
  }
}
