// The jump: NVIDIA's jump_0_5m smart primitive (assets/smart_primitives/soma) in a canonical frame
// (the character at the origin, facing +Z), each target a four-pose window the model generates toward.

import { canonicalWindows, type Move, type Rig } from './locomotion.ts'
import type { Clip } from './skeleton.ts'

/** Both variations of the primitive, minus the run-up step: the pose over the hurdle, then the landing. */
const JUMP_CLIPS = [
  { file: 'jump_over_obstacle_0_5m_R_002__A437.bvh', times: [2.16, 2.7] },
  { file: 'jump_over_obstacle_0_5m_R_001__A415.bvh', times: [3.25, 3.87] },
]
/** The hurdle step hands over 1 frame early, the landing 8, as the primitive sets them. */
const JUMP_HANDOVER = [1, 8]
const JUMP_TOKENS = [6, 7, 8, 9, 10, 11, 12, 13, 14]

/**
 * Jumps alternate between the variations, whose clips `load` reads by file name; they start further
 * ahead at speed, so the run-up isn't squeezed.
 */
export function jumps(rig: Rig, load: (file: string) => Clip): () => Move {
  const variations: Move[] = JUMP_CLIPS.map(({ file, times }) => ({
    steps: canonicalWindows(rig, load(file), times).map((window, k) => ({
      window,
      handover: JUMP_HANDOVER[k],
      tokens: JUMP_TOKENS,
    })),
    lead: pace => 1.5 + 0.25 * pace,
  }))
  let next = 0
  return () => variations[next++ % variations.length]
}
