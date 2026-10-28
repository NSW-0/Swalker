import { useEffect, useState } from 'react'

const FRAME_MS = 140

function prefersReducedMotion() {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

// Plays a walk's silhouettes as a loop, with the filmstrip underneath. The frames are the real
// 64x44 images GLTNet reads, so they are scaled up with crisp pixels.
export default function WalkPlayer({ frames, label, title }: { frames: string[]; label: string; title: string }) {
  const [index, setIndex] = useState(0)

  useEffect(() => {
    if (frames.length < 2 || prefersReducedMotion()) return
    const id = window.setInterval(() => setIndex((i) => (i + 1) % frames.length), FRAME_MS)
    return () => window.clearInterval(id)
  }, [frames.length])

  const current = frames.length ? frames[index % frames.length] : null

  return (
    <figure className="walk-player">
      <figcaption>
        <span className="walk-player-label">{label}</span>
        <span className="walk-player-title" title={title}>
          {title}
        </span>
      </figcaption>
      <div className="walk-player-screen">
        {current ? <img className="sil-img" src={current} alt={`${title}, walking silhouette`} /> : <span>No silhouettes on file</span>}
      </div>
      {frames.length > 0 && (
        <div className="walk-player-strip" aria-hidden="true">
          {frames.map((src, i) => (
            <img key={i} className={i === index % frames.length ? 'sil-img is-current' : 'sil-img'} src={src} alt="" />
          ))}
        </div>
      )}
    </figure>
  )
}

export function EmptyCaseFile({ children }: { children: React.ReactNode }) {
  return (
    <figure className="walk-player walk-player-empty">
      <figcaption>
        <span className="walk-player-label">On file</span>
        <span className="walk-player-title">No case file</span>
      </figcaption>
      <div className="walk-player-screen">
        <span className="walk-player-question" aria-hidden="true">
          ?
        </span>
      </div>
      <div className="walk-player-empty-body">{children}</div>
    </figure>
  )
}
