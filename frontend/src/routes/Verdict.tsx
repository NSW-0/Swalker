import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogTitle from '@mui/material/DialogTitle'
import DialogContent from '@mui/material/DialogContent'
import DialogActions from '@mui/material/DialogActions'
import TextField from '@mui/material/TextField'
import Tooltip from '@mui/material/Tooltip'
import ArchiveOutlinedIcon from '@mui/icons-material/ArchiveOutlined'
import CheckCircleOutlinedIcon from '@mui/icons-material/CheckCircleOutlined'
import InvestigationLoader from '../components/verdict/InvestigationLoader'
import { SilhouetteStrip, SilhouetteThumb } from '../components/verdict/MockSilhouette'
import { buildMockVerdict, closeness, MATCH_THRESHOLD, SEQ_LEN } from '../data/mockVerdict'
import type { MockMode, Suspect } from '../data/mockVerdict'
import { FEATURE_DIM } from '../data/mockGallery'
import { isNameTaken, nextPersonId, saveToArchive } from '../data/archiveStore'
import duckGem from '../assets/duckGem.png'
// Web-sized copies: duckNoGem.png is 6.5 MB, wing.png has a lot of empty space.
import duckNoGem from '../assets/duckNoGem_web.webp'
import wingMarker from '../assets/wing_web.png'

// Nothing is computed yet, so the loading screen just runs for a fixed time.
const LOADING_MS = 3500

function SuspectCard({ suspect, rank, isPrime }: { suspect: Suspect; rank: number; isPrime: boolean }) {
  return (
    <div className={isPrime ? 'suspect-card suspect-card-prime' : 'suspect-card'}>
      <div className="suspect-card-thumb">
        <SilhouetteThumb id={suspect.personId} />
      </div>
      <div>
        <p className="suspect-card-rank">
          #{rank}
          {isPrime && ' · prime suspect'}
        </p>
        <p className="suspect-card-id">{suspect.personId}</p>
        <p className="suspect-card-dist">distance {suspect.distance.toFixed(2)}</p>
      </div>
      <p className="suspect-card-conf">
        {Math.round(suspect.confidence)}%<span>strength</span>
      </p>
    </div>
  )
}

export default function Verdict() {
  const [searchParams] = useSearchParams()
  // Dev helper until the backend exists. No match is the default for now;
  // /crime-scene/verdict?mock=match shows the match result.
  const mode: MockMode = searchParams.get('mock') === 'match' ? 'match' : 'unknown'
  const verdict = useMemo(() => buildMockVerdict(mode), [mode])
  const [loading, setLoading] = useState(true)

  // Saving an unknown walker to the Archive
  const [saveOpen, setSaveOpen] = useState(false)
  const [newName, setNewName] = useState('')
  const [savedAs, setSavedAs] = useState<string | null>(null)
  const trimmedName = newName.trim()
  const nameTaken = trimmedName !== '' && isNameTaken(trimmedName)

  const openSaveDialog = () => {
    setNewName(nextPersonId())
    setSaveOpen(true)
  }

  const handleSave = () => {
    if (trimmedName === '' || nameTaken) return
    saveToArchive({
      personId: trimmedName,
      videoPath: `uploads/${verdict.clipName}`,
      featureVector: verdict.queryVector,
    })
    setSavedAs(trimmedName)
    setSaveOpen(false)
  }

  const best = verdict.suspects[0]
  const thresholdPos = closeness(MATCH_THRESHOLD) * 100

  return (
    <section className="page page-verdict-content">
      <Tooltip
        title="Back to investigate more clues"
        placement="right"
        slotProps={{
          tooltip: {
            sx: {
              bgcolor: 'rgba(54, 53, 58, 0.97)',
              border: '1px solid var(--border)',
              fontSize: '13px',
              fontWeight: 600,
              boxShadow: '0 6px 16px rgba(0, 0, 0, 0.35)',
            },
          },
        }}
      >
        <Link to="/crime-scene" className="verdict-back" aria-label="Back to investigate more clues">
          <svg viewBox="0 0 48 48" aria-hidden="true">
            <defs>
              <linearGradient id="verdict-back-gradient" gradientUnits="userSpaceOnUse" x1="0" y1="3" x2="0" y2="45">
                <stop offset="0%" stopColor="#FBAA2C" />
                <stop offset="100%" stopColor="#E8443A" />
              </linearGradient>
            </defs>
            <g fill="none" stroke="url(#verdict-back-gradient)" strokeWidth="3.4" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="24" cy="24" r="21.3" />
              <path d="M33 24H15M22.5 16.5L15 24l7.5 7.5" />
            </g>
          </svg>
        </Link>
      </Tooltip>

      <div className="verdict-header">
        <h1 className="verdict-heading">{loading ? 'Investigating…' : 'The Verdict'}</h1>
        <p className="verdict-caseline">
          <span className="verdict-tag">CASE #{verdict.caseId}</span>
          evidence: {verdict.clipName}
          <span className="verdict-tag">compared against {verdict.gallerySize} case files</span>
        </p>
      </div>

      {loading ? (
        <InvestigationLoader duration={LOADING_MS} onDone={() => setLoading(false)} />
      ) : (
        <div className="verdict-stage">
          <div className="verdict-grid">
            <article className="verdict-bubble">
              <span className="verdict-bubble-tab">{verdict.isMatch ? 'Prime suspect' : 'No match'}</span>

              <div className="verdict-suspect-head">
                <div>
                  <h2 className="verdict-suspect-id">{verdict.isMatch ? best.personId : 'Unknown walker'}</h2>
                  <p className="verdict-suspect-sub">
                    {verdict.isMatch ? 'Closest walk in the Archive' : "This walk isn't in the Archive yet."}
                  </p>
                </div>
                <span className={verdict.isMatch ? 'verdict-stamp' : 'verdict-stamp verdict-stamp-unknown'}>
                  {verdict.isMatch ? 'Match found' : 'Unidentified'}
                </span>
              </div>

              <div className="verdict-meter">
                <div className="verdict-meter-label">
                  <span>{verdict.isMatch ? 'How close the walks are' : `Closest walk: ${best.personId}`}</span>
                  <span>distance {best.distance.toFixed(2)}</span>
                </div>
                <div className="verdict-meter-track">
                  <div
                    className={verdict.isMatch ? 'verdict-meter-fill' : 'verdict-meter-fill verdict-meter-fill-weak'}
                    style={{ width: `${closeness(best.distance) * 100}%` }}
                  />
                  <div className="verdict-meter-threshold" style={{ left: `${thresholdPos}%` }}>
                    <img src={wingMarker} alt="" />
                    <span>match line</span>
                  </div>
                </div>
              </div>

              <div className="verdict-evidence">
                <div>
                  <p className="verdict-strip-label">The clue · {verdict.clipName}</p>
                  <SilhouetteStrip id={verdict.isMatch ? best.personId : verdict.clipName} />
                </div>
                {verdict.isMatch && (
                  <div>
                    <p className="verdict-strip-label">On file · {best.personId}</p>
                    <SilhouetteStrip id={best.personId} offset={0.3} />
                  </div>
                )}
              </div>

              {!verdict.isMatch && (
                <div className="verdict-actions">
                  {savedAs ? (
                    <p className="verdict-saved">
                      <CheckCircleOutlinedIcon fontSize="small" />
                      Saved to the Archive as <b>{savedAs}</b>
                      <Link to={`/archive?person=${encodeURIComponent(savedAs)}`}>See it in the Archive →</Link>
                    </p>
                  ) : (
                    <Button
                      variant="contained"
                      startIcon={<ArchiveOutlinedIcon />}
                      onClick={openSaveDialog}
                      sx={{ borderRadius: '999px', px: 3, py: 1.25 }}
                    >
                      Add to the Archive
                    </Button>
                  )}
                </div>
              )}

              <p className="verdict-fine">
                {verdict.isMatch
                  ? 'Match strength compares the suspects in the lineup with each other. It is not an absolute certainty.'
                  : savedAs
                    ? 'Next time this person walks past a camera, Swalker can recognise them.'
                    : 'Nobody in the Archive walks close enough to cross the match line. Save this walk so Swalker recognises them next time.'}
              </p>
            </article>

            <aside className="verdict-side">
              <h2 className="verdict-side-title">{verdict.isMatch ? 'The Lineup' : 'Nearest walks'}</h2>
              <p className="verdict-side-sub">
                {verdict.isMatch ? 'Top 3 closest walks, ranked by distance' : 'None of these crossed the match line'}
              </p>
              <div className="verdict-lineup">
                {verdict.suspects.map((s, i) => (
                  <SuspectCard key={s.personId} suspect={s} rank={i + 1} isPrime={verdict.isMatch && i === 0} />
                ))}
              </div>

              <h2 className="verdict-side-title verdict-trail-title">How we got here</h2>
              <ol className="verdict-trail">
                <li>
                  <b>1 person</b> spotted in {verdict.framesWithPerson} frames
                </li>
                <li>
                  <b>{SEQ_LEN} silhouettes</b> sampled from the clip
                </li>
                <li>
                  <b>{FEATURE_DIM}-number</b> gait signature built
                </li>
                <li>
                  <b>{verdict.gallerySize} case files</b> compared
                </li>
              </ol>
            </aside>
          </div>

          {verdict.isMatch ? (
            <img src={duckGem} className="verdict-duck verdict-duck-gem" alt="" />
          ) : (
            <img src={duckNoGem} className="verdict-duck verdict-duck-nogem" alt="" />
          )}
        </div>
      )}

      <Dialog
        open={saveOpen}
        onClose={() => setSaveOpen(false)}
        maxWidth="xs"
        fullWidth
        slotProps={{
          paper: {
            sx: {
              background: 'rgba(54, 53, 58, 0.97)',
              backgroundImage: 'none',
              border: '1px solid var(--border)',
              borderRadius: '12px',
            },
          },
        }}
      >
        <form
          onSubmit={(e) => {
            e.preventDefault()
            handleSave()
          }}
        >
          <DialogTitle sx={{ fontWeight: 700 }}>Add to the Archive</DialogTitle>
          <DialogContent>
            <p className="verdict-dialog-text">
              Open a new case file for this walker. Their walk from {verdict.clipName} is saved under this name.
            </p>
            <TextField
              autoFocus
              fullWidth
              label="Case file name"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              error={nameTaken}
              helperText={nameTaken ? 'A case file with this name already exists.' : ' '}
              sx={{ mt: 1 }}
            />
          </DialogContent>
          <DialogActions sx={{ px: 3, pb: 2.5 }}>
            <Button onClick={() => setSaveOpen(false)} sx={{ color: 'var(--muted)' }}>
              Cancel
            </Button>
            <Button
              type="submit"
              variant="contained"
              disabled={trimmedName === '' || nameTaken}
              sx={{ borderRadius: '999px', px: 3 }}
            >
              Save
            </Button>
          </DialogActions>
        </form>
      </Dialog>
    </section>
  )
}
