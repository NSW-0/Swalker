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

// The server doesn't report progress yet, so while waiting the ring eases towards WAIT_CAP
// (40% at 3.5 s, 73% at 10 s, 85% at 15 s) and only completes once the answer is in.
const WAIT_CAP = 0.9
const WAIT_TAU = 6000
const FINISH_MS = 600

interface Props {
  done: boolean // the answer (or an error) has arrived
  minDuration?: number // keep the investigation on screen at least this long
  onDone: () => void
}

export default function InvestigationLoader({ done, minDuration = 3500, onDone }: Props) {
  const [progress, setProgress] = useState(0)
  const onDoneRef = useRef(onDone)
  const doneRef = useRef(done)

  useEffect(() => {
    onDoneRef.current = onDone
    doneRef.current = done
  }, [onDone, done])

  useEffect(() => {
    let raf = 0
    const start = performance.now()
    let finishFrom: { t: number; p: number } | null = null
    let current = 0

    const tick = (now: number) => {
      const elapsed = now - start
      const waiting = WAIT_CAP * (1 - Math.exp(-elapsed / WAIT_TAU))
      if (!finishFrom && doneRef.current && elapsed >= minDuration) finishFrom = { t: now, p: current }
      current = finishFrom ? finishFrom.p + (1 - finishFrom.p) * Math.min(1, (now - finishFrom.t) / FINISH_MS) : waiting
      setProgress(current)

      if (current < 1) raf = requestAnimationFrame(tick)
      else onDoneRef.current()
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [minDuration])

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
