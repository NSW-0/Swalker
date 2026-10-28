import { scalePosition } from '../../api/client'
import wingMarker from '../../assets/wing_web.png'

// Left = identical walks, the wing = the match line, right = strangers. The pin is this walk.
export default function DistanceScale({ distance, threshold, isMatch }: { distance: number; threshold: number; isMatch: boolean }) {
  const pin = scalePosition(distance, threshold) * 100
  return (
    <div className="distance-scale">
      <div className="distance-scale-track">
        <div className="distance-scale-zone" />
        <div className="distance-scale-line">
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
        <span>match line · {threshold}</span>
        <span>strangers</span>
      </div>
    </div>
  )
}
