// Downloads the MotionBricks SOMA model and the sample clips this demo steers with, straight from
// NVIDIA's repository at a pinned commit, verifying each Git LFS object's SHA-256.
//
//   bun run fetch-assets          # mixed-FP16 model (380 MB)
//   bun run fetch-assets --fp32   # FP32 model too (758 MB)

import { createHash } from 'node:crypto'
import { createReadStream, createWriteStream, existsSync, mkdirSync, renameSync, rmSync, statSync } from 'node:fs'
import { dirname, join, relative } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'

import { ASSETS_DIR, MANIFEST_FILE, modelFile } from '../src/assets.ts'

const REPO = 'NVlabs/motionbricks'
const COMMIT = '54fc563aeaaf819a753d72cc9fdc6713989cdd43'
const RAW = `https://raw.githubusercontent.com/${REPO}/${COMMIT}/`
const MEDIA = `https://media.githubusercontent.com/media/${REPO}/${COMMIT}/`
const SOMA = 'motionbricks/assets/onnx/soma/'

/** Repository path → local name under assets/. */
const FILES: Array<{ src: string; dest: string; lfs: boolean }> = [
  { src: SOMA + MANIFEST_FILE, dest: MANIFEST_FILE, lfs: false },
  { src: SOMA + modelFile(), dest: modelFile(), lfs: true },
  { src: 'motionbricks/assets/onnx/LICENSE.md', dest: 'licenses/MODEL_LICENSE.md', lfs: false },
  { src: 'assets/motions/LICENSE.txt', dest: 'licenses/SAMPLE_MOTIONS_LICENSE.txt', lfs: false },
  ...[
    'locomotion/idle/idle_loop_001__A268',
    'locomotion/bs_1d_linear/Neutral_walk_forward_002__A057',
    'locomotion/bs_1d_linear/neutral_jog_ff_180_R_001__A534',
    'locomotion/bs_1d_linear/run_loop_180_R_003__A326',
    'jump/jump_over_obstacle_0_5m_R_002__A437',
    'jump/jump_over_obstacle_0_5m_R_001__A415',
  ].map(clip => ({
    src: `assets/motions/soma/${clip}.bvh`,
    dest: `motions/${clip.split('/').at(-1)}.bvh`,
    lfs: true,
  })),
]
if (process.argv.includes('--fp32')) {
  FILES.push({ src: SOMA + modelFile(true), dest: modelFile(true), lfs: true })
}

async function text(url: string): Promise<string> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`)
  return res.text()
}

async function sha256(path: string): Promise<string> {
  const hash = createHash('sha256')
  await pipeline(createReadStream(path), hash)
  return hash.digest('hex')
}

async function download(url: string, path: string, size: number): Promise<void> {
  const res = await fetch(url)
  if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`)
  let done = 0
  let shown = -1
  const body = Readable.fromWeb(res.body)
  body.on('data', (chunk: Buffer) => {
    done += chunk.length
    const pct = Math.floor((done / size) * 100)
    if (size > 5e6 && pct !== shown) {
      shown = pct
      process.stdout.write(`\r  ${(done / 1e6).toFixed(0)} / ${(size / 1e6).toFixed(0)} MB`)
    }
  })
  await pipeline(body, createWriteStream(path))
  if (size > 5e6) process.stdout.write('\n')
}

async function fetchLfs(src: string, dest: string): Promise<void> {
  const pointer = await text(RAW + src)
  const oid = /oid sha256:([0-9a-f]{64})/.exec(pointer)?.[1]
  const size = Number(/size (\d+)/.exec(pointer)?.[1])
  if (!oid || !Number.isFinite(size)) throw new Error(`${src}: not a Git LFS pointer`)
  if (existsSync(dest) && statSync(dest).size === size && (await sha256(dest)) === oid) {
    console.log(`✓ ${relative(process.cwd(), dest)} (cached)`)
    return
  }
  console.log(`↓ ${src}`)
  const partial = `${dest}.partial`
  await download(MEDIA + src, partial, size)
  const got = await sha256(partial)
  if (got !== oid) {
    rmSync(partial, { force: true })
    throw new Error(`${src}: SHA-256 mismatch (${got})`)
  }
  renameSync(partial, dest)
  console.log(`✓ ${relative(process.cwd(), dest)}`)
}

console.log(`MotionBricks assets from github.com/${REPO} @ ${COMMIT.slice(0, 7)}\n`)
for (const { src, dest, lfs } of FILES) {
  const path = join(ASSETS_DIR, dest)
  mkdirSync(dirname(path), { recursive: true })
  if (lfs) await fetchLfs(src, path)
  else {
    await download(RAW + src, path, 0)
    console.log(`✓ ${relative(process.cwd(), path)}`)
  }
}
console.log(`
These files are NVIDIA's and stay on this machine (assets/ is git-ignored):
  • the model: NVIDIA Open Model Agreement (assets/licenses/MODEL_LICENSE.md)
  • the .bvh clips: NVIDIA Sample Data Evaluation License — internal evaluation only,
    no redistribution (assets/licenses/SAMPLE_MOTIONS_LICENSE.txt)`)
