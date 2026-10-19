import { useEffect, useRef, useState } from 'react'
import spriteFrames from '../../assets/sprite_frames.png'

// sprite_frames.png is sprite.png cut into 8 equal cells in one row,
// in story order: develop the photo, hang it up, inspect it, then relax.
const FRAME_COUNT = 8

const STEPS = [
  'Spotting the suspect',
  'Lifting the silhouettes',
  'Reading the walk',
  'Searching the Archive',
]

const SIZE = 300
const STROKE = 10
const RADIUS = (SIZE - STROKE) / 2
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

interface Props {
  duration: number
  onDone: () => void
}

export default function InvestigationLoader({ duration, onDone }: Props) {
  const [progress, setProgress] = useState(0)
  const onDoneRef = useRef(onDone)

  useEffect(() => {
    onDoneRef.current = onDone
  }, [onDone])

  useEffect(() => {
    let raf = 0
    const start = performance.now()
    const tick = (now: number) => {
      const p = Math.min((now - start) / duration, 1)
      setProgress(p)
      if (p < 1) raf = requestAnimationFrame(tick)
      else onDoneRef.current()
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [duration])

  const frame = Math.min(FRAME_COUNT - 1, Math.floor(progress * FRAME_COUNT))
  const step = Math.min(STEPS.length - 1, Math.floor(progress * STEPS.length))
  const percent = Math.round(progress * 100)

  return (
    <div className="loader">
      <div
        className="loader-ring"
        role="progressbar"
        aria-label="Investigating the clue"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
      >
        <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`} aria-hidden="true">
          <circle className="loader-track" cx={SIZE / 2} cy={SIZE / 2} r={RADIUS} strokeWidth={STROKE} />
          <circle
            className="loader-fill"
            cx={SIZE / 2}
            cy={SIZE / 2}
            r={RADIUS}
            strokeWidth={STROKE}
            strokeDasharray={CIRCUMFERENCE}
            strokeDashoffset={CIRCUMFERENCE * (1 - progress)}
          />
        </svg>
        <div className="loader-disc">
          <div
            className="loader-sprite"
            style={{
              backgroundImage: `url(${spriteFrames})`,
              backgroundSize: `${FRAME_COUNT * 100}% 100%`,
              backgroundPosition: `${(frame / (FRAME_COUNT - 1)) * 100}% 0`,
            }}
          />
        </div>
      </div>

      <p className="loader-step">{STEPS[step]}…</p>
      <p className="loader-percent">{percent}%</p>
    </div>
  )
}
