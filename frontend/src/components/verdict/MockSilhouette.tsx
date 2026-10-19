// Placeholder silhouettes drawn as SVG, 44x64 like the model's input.
// Once the backend exists these get replaced by the real binary masks that
// silhouette_extractor.py produces (sent as PNG images), so they will look
// like rough white blobs of the actual person, not clean stick figures.

interface Build {
  stride: number
  torsoW: number
  legLen: number
}

// Stable body proportions per id, so each mock person always looks the same.
function buildFor(id: string): Build {
  let h = 0
  for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) | 0
  const r = (n: number) => ((Math.abs(h) >> n) % 100) / 100
  return { stride: 20 + r(1) * 12, torsoW: 8 + r(3) * 3.5, legLen: 12 + r(5) * 2 }
}

export function MockSilhouette({ phase, build }: { phase: number; build: Build }) {
  const { stride, torsoW, legLen } = build
  const s = Math.sin(phase)
  const c = Math.cos(phase)
  const bob = Math.abs(c) * 1.2
  const hipY = 37 - bob
  const shY = 18 - bob

  const leg = (sign: number) => {
    const a = ((stride * sign * s) * Math.PI) / 180
    const kx = 22 + Math.sin(a) * legLen
    const ky = hipY + Math.cos(a) * legLen
    const b = a - Math.max(0, sign * c) * 0.6
    const fx = kx + Math.sin(b) * legLen
    const fy = ky + Math.cos(b) * legLen
    return `M22 ${hipY} L${kx.toFixed(1)} ${ky.toFixed(1)} L${fx.toFixed(1)} ${fy.toFixed(1)}`
  }
  const arm = (sign: number) => {
    const a = ((-22 * sign * s) * Math.PI) / 180
    return `M22 ${shY} L${(22 + Math.sin(a) * 15).toFixed(1)} ${(shY + Math.cos(a) * 15).toFixed(1)}`
  }

  return (
    <svg viewBox="0 0 44 64" aria-hidden="true">
      <rect width="44" height="64" fill="#000" />
      <g stroke="#fff" strokeLinecap="round" strokeLinejoin="round" fill="none">
        <path d={arm(-1)} strokeWidth={3.6} />
        <path d={leg(-1)} strokeWidth={5} />
        <path d={`M22 ${16 - bob} L22 ${hipY}`} strokeWidth={torsoW} />
        <path d={leg(1)} strokeWidth={5} />
        <path d={arm(1)} strokeWidth={3.6} />
      </g>
      <circle cx="22" cy={(9 - bob).toFixed(1)} r="5.4" fill="#fff" />
    </svg>
  )
}

export function SilhouetteThumb({ id }: { id: string }) {
  return <MockSilhouette phase={1.4} build={buildFor(id)} />
}

export function SilhouetteStrip({
  id,
  frames = 10,
  offset = 0,
}: {
  id: string
  frames?: number
  offset?: number
}) {
  const build = buildFor(id)
  return (
    <div className="verdict-strip">
      {Array.from({ length: frames }, (_, i) => (
        <MockSilhouette key={i} phase={(i / frames) * Math.PI * 2 + offset} build={build} />
      ))}
    </div>
  )
}
