import { scalePosition } from '../../api/client'
import wingMarker from '../../assets/wing_web.png'

// Left = identical walks, the wing = the match line, right = strangers. The pin is this walk.
export default function DistanceScale({
  distance,
  threshold,
  end,
  isMatch,
}: {
  distance: number
  threshold: number
  end: number
  isMatch: boolean
}) {
  const line = scalePosition(threshold, end) * 100
  // Kept a little inside the ends, so the pin and its label never hang off the bar.
  const pin = Math.max(3, Math.min(97, scalePosition(distance, end) * 100))
  return (
    <div className="distance-scale">
      <div className="distance-scale-track">
        <div className="distance-scale-zone" style={{ right: `${100 - line}%` }} />
        <div className="distance-scale-line" style={{ left: `${line}%` }}>
          <img src={wingMarker} alt="" />
        </div>
        <div
          className={isMatch ? 'distance-scale-pin' : 'distance-scale-pin distance-scale-pin-out'}
          style={{ left: `${pin}%` }}
          role="img"
          aria-label={`distance ${distance.toFixed(1)}, match line at ${threshold}`}
        >
          <span>{distance.toFixed(1)}</span>
        </div>
      </div>
      <div className="distance-scale-labels">
        <span>identical</span>
        <span style={{ left: `${line}%` }}>match line · {threshold}</span>
        <span>far apart · {end}</span>
      </div>
    </div>
  )
}
