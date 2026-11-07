/** [x, y, z] */
export type Vec3 = [number, number, number]
/** Unit quaternion [x, y, z, w]. */
export type Quat = [number, number, number, number]

export const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
export const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
export const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s]
export const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
export const length = (a: Vec3): number => Math.hypot(a[0], a[1], a[2])
export const lerp3 = (a: Vec3, b: Vec3, t: number): Vec3 => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
]

export function normalize(a: Vec3): Vec3 {
  const n = length(a)
  return n > 1e-12 ? scale(a, 1 / n) : [0, 0, 0]
}

/** The XZ part of a point, on the ground. */
export const ground = (p: Vec3): Vec3 => [p[0], 0, p[2]]

export const identity = (): Quat => [0, 0, 0, 1]

export function qmul(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a
  const [bx, by, bz, bw] = b
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ]
}

/** Inverse of a unit quaternion. */
export const qinv = (q: Quat): Quat => [-q[0], -q[1], -q[2], q[3]]

export function qnormalize(q: Quat): Quat {
  const n = Math.hypot(q[0], q[1], q[2], q[3])
  return n > 1e-12 ? [q[0] / n, q[1] / n, q[2] / n, q[3] / n] : identity()
}

export function qrotate(q: Quat, v: Vec3): Vec3 {
  const u: Vec3 = [q[0], q[1], q[2]]
  const t = scale(cross(u, v), 2)
  return add(add(v, scale(t, q[3])), cross(u, t))
}

export function qaxisAngle(axis: Vec3, angle: number): Quat {
  const s = Math.sin(angle / 2)
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)]
}

/** Rotation about +Y; a yaw of `h` turns +Z to face heading `h`. */
export const qyaw = (angle: number): Quat => qaxisAngle([0, 1, 0], angle)

export function qslerp(a: Quat, b: Quat, t: number): Quat {
  let [bx, by, bz, bw] = b
  let cos = a[0] * bx + a[1] * by + a[2] * bz + a[3] * bw
  if (cos < 0) {
    cos = -cos
    bx = -bx
    by = -by
    bz = -bz
    bw = -bw
  }
  if (cos > 0.9995) {
    return qnormalize([a[0] + (bx - a[0]) * t, a[1] + (by - a[1]) * t, a[2] + (bz - a[2]) * t, a[3] + (bw - a[3]) * t])
  }
  const theta = Math.acos(cos)
  const sin = Math.sin(theta)
  const wa = Math.sin((1 - t) * theta) / sin
  const wb = Math.sin(t * theta) / sin
  return [a[0] * wa + bx * wb, a[1] * wa + by * wb, a[2] * wa + bz * wb, a[3] * wa + bw * wb]
}

/** Shortest-arc rotation taking unit vector `from` onto unit vector `to`. */
export function qfromTo(from: Vec3, to: Vec3): Quat {
  const d = dot(from, to)
  if (d < -0.999999) {
    const axis = length(cross([1, 0, 0], from)) > 1e-6 ? cross([1, 0, 0], from) : cross([0, 1, 0], from)
    return qaxisAngle(normalize(axis), Math.PI)
  }
  const c = cross(from, to)
  return qnormalize([c[0], c[1], c[2], 1 + d])
}

/** Rotation whose matrix has the orthonormal columns x, y, z. */
export function qfromBasis(x: Vec3, y: Vec3, z: Vec3): Quat {
  const [m00, m10, m20] = x
  const [m01, m11, m21] = y
  const [m02, m12, m22] = z
  const trace = m00 + m11 + m22
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2
    return qnormalize([(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, s / 4])
  }
  if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2
    return qnormalize([s / 4, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s])
  }
  if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2
    return qnormalize([(m01 + m10) / s, s / 4, (m12 + m21) / s, (m02 - m20) / s])
  }
  const s = Math.sqrt(1 + m22 - m00 - m11) * 2
  return qnormalize([(m02 + m20) / s, (m12 + m21) / s, s / 4, (m10 - m01) / s])
}

/** Wrap an angle to [-π, π]. */
export const wrapAngle = (a: number): number => Math.atan2(Math.sin(a), Math.cos(a))

/** Unit ground direction a heading faces. */
export const headingDir = (h: number): Vec3 => [Math.sin(h), 0, Math.cos(h)]

/** The heading a direction faces along the ground: the inverse of headingDir. */
export const headingFromDir = (dir: Vec3): number => Math.atan2(dir[0], dir[2])

/**
 * A pose's heading from its hip joints, the model's training convention ("hips_pos"):
 * 0 faces +Z, π/2 faces +X. Facing +Z puts the left hip at +X.
 */
export function headingFromHips(rightHip: Vec3, leftHip: Vec3): number {
  const across: Vec3 = [rightHip[0] - leftHip[0], 0, rightHip[2] - leftHip[2]]
  const n = length(across)
  if (n <= 1e-6) return 0
  return headingFromDir(cross([0, 1, 0], scale(across, 1 / n)))
}
