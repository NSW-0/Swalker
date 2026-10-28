import { useEffect, useRef, useState } from 'react'
import type { JobStage } from '../../api/client'
import spriteFrames from '../../assets/sprite_frames.png'

// sprite_frames.png is sprite.png cut into 8 equal cells in one row,
// in story order: develop the photo, hang it up, inspect it, then relax.
const FRAME_COUNT = 8

// One label per server stage (detecting, silhouettes, embedding, searching).
const STEPS = [
  'Spotting the suspect',
  'Lifting the silhouettes',
  'Reading the walk',
  'Searching the Archive',
]
const STEP_OF: Partial<Record<JobStage, number>> = {
  queued: 0,
  detecting: 0,
  silhouettes: 1,
  embedding: 2,
  searching: 3,
  done: 3,
}

const SIZE = 300
const STROKE = 10
const RADIUS = (SIZE - STROKE) / 2
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

const EASE_MS = 250 // the ring glides towards the server's progress instead of jumping
const STEP_MIN_MS = 450 // each label stays readable, even when the server passes a stage in 20 ms
const FINISH_MS = 500
// Without a status (not reported yet), ease towards WAIT_CAP over time.
const WAIT_CAP = 0.9
const WAIT_TAU = 6000

interface Props {
  done: boolean // the answer (or an error) has arrived
  stage: JobStage | null // latest stage reported by the server
  progress: number | null // latest overall progress reported by the server, 0..1
  minDuration?: number // keep the investigation on screen at least this long
  onDone: () => void
}

export default function InvestigationLoader({ done, stage, progress, minDuration = 3500, onDone }: Props) {
  const [view, setView] = useState({ progress: 0, step: 0 })
  const live = useRef({ done, stage, progress, onDone })

  useEffect(() => {
    live.current = { done, stage, progress, onDone }
  }, [done, stage, progress, onDone])

  useEffect(() => {
    let raf = 0
    const start = performance.now()
    let last = start
    let shown = 0
    let step = 0
    let stepAt = start
    let finishFrom: { t: number; p: number } | null = null

    const tick = (now: number) => {
      const { done: isDone, stage: s, progress: p } = live.current
      const elapsed = now - start
      const dt = now - last
      last = now

      // Labels follow the server's stage, one step at a time.
      const serverStep = isDone ? STEPS.length - 1 : s ? (STEP_OF[s] ?? step) : Math.floor(shown * STEPS.length * 0.99)
      if (serverStep > step && now - stepAt >= STEP_MIN_MS) {
        step += 1
        stepAt = now
      }

      if (!finishFrom && isDone && elapsed >= minDuration && step === STEPS.length - 1) {
        finishFrom = { t: now, p: shown }
      }
      if (finishFrom) {
        shown = finishFrom.p + (1 - finishFrom.p) * Math.min(1, (now - finishFrom.t) / FINISH_MS)
      } else {
        const goal = p ?? WAIT_CAP * (1 - Math.exp(-elapsed / WAIT_TAU))
        // Glide, never backwards, and keep a little for the finish.
        shown = Math.max(shown, shown + (Math.min(goal, 0.97) - shown) * Math.min(1, dt / EASE_MS))
      }

      setView({ progress: shown, step })
      if (shown < 1) raf = requestAnimationFrame(tick)
      else live.current.onDone()
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [minDuration])

  const frame = Math.min(FRAME_COUNT - 1, Math.floor(view.progress * FRAME_COUNT))
  const percent = Math.round(view.progress * 100)

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
            strokeDashoffset={CIRCUMFERENCE * (1 - view.progress)}
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

      <p className="loader-step">{stage === 'queued' ? 'Waiting for the detective' : STEPS[view.step]}…</p>
      <p className="loader-percent">{percent}%</p>
    </div>
  )
}
