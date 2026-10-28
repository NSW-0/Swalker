import { useEffect, useState } from 'react'
import { Link, Navigate } from 'react-router-dom'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogTitle from '@mui/material/DialogTitle'
import DialogContent from '@mui/material/DialogContent'
import DialogActions from '@mui/material/DialogActions'
import TextField from '@mui/material/TextField'
import Tooltip from '@mui/material/Tooltip'
import ArchiveOutlinedIcon from '@mui/icons-material/ArchiveOutlined'
import CheckCircleOutlinedIcon from '@mui/icons-material/CheckCircleOutlined'
import SearchOutlinedIcon from '@mui/icons-material/SearchOutlined'
import InvestigationLoader from '../components/verdict/InvestigationLoader'
import DistanceScale from '../components/verdict/DistanceScale'
import WalkPlayer, { EmptyCaseFile } from '../components/verdict/WalkPlayer'
import { SilhouetteThumb } from '../components/verdict/Silhouettes'
import { api, ApiError, FEATURE_DIM, scalePosition } from '../api/client'
import type { JobStatus, Suspect, VerdictResult } from '../api/client'
import { getClue, investigateClue } from '../data/clueStore'
import duckGem from '../assets/duckGem.png'
// Web-sized copy: duckNoGem.png is 6.5 MB.
import duckNoGem from '../assets/duckNoGem_web.webp'

function SuspectCard({ suspect, rank, threshold, isPrime }: { suspect: Suspect; rank: number; threshold: number; isPrime: boolean }) {
  const inside = suspect.distance <= threshold
  return (
    <div className={isPrime ? 'suspect-card suspect-card-prime' : 'suspect-card'}>
      <div className="suspect-card-thumb">
        <SilhouetteThumb src={suspect.thumbnailUrl} />
      </div>
      <div className="suspect-card-body">
        <p className="suspect-card-rank">
          #{rank}
          {isPrime && ' · prime suspect'}
        </p>
        <p className="suspect-card-id" title={suspect.personId}>
          {suspect.personId}
        </p>
        <div className="suspect-card-bar" aria-hidden="true">
          <span className={inside ? 'is-inside' : ''} style={{ width: `${(1 - scalePosition(suspect.distance, threshold)) * 100}%` }} />
          <i />
        </div>
        <p className="suspect-card-dist">
          distance <b>{suspect.distance.toFixed(1)}</b>
          <span title="Relative between these suspects only, not a probability">
            {Math.round(suspect.confidence)}% strength
          </span>
        </p>
      </div>
    </div>
  )
}

function suggestedName(gallerySize: number) {
  return `person_${String(gallerySize + 1).padStart(3, '0')}`
}

// Three numbers that explain the decision. For a match, the gap to the runner-up shows how clear it is.
function verdictFacts(v: VerdictResult, best: Suspect, matched: boolean): [string, string][] {
  const second = v.suspects[1]
  if (matched) {
    return [
      ['Distance', best.distance.toFixed(1)],
      ['Match line', String(v.threshold)],
      ['Lead over #2', second ? (second.distance - best.distance).toFixed(1) : 'no #2'],
    ]
  }
  return [
    ['Nearest', best.distance.toFixed(1)],
    ['Match line', String(v.threshold)],
    ['Past the line by', (best.distance - v.threshold).toFixed(1)],
  ]
}

function verdictSentence(v: VerdictResult, best: Suspect | undefined) {
  if (!best) return 'The Archive is empty, so there is no one to compare this walk with yet.'
  const apart = best.distance.toFixed(1)
  if (v.isMatch) {
    return best.distance <= v.threshold / 2
      ? `The walks are ${apart} apart, well inside the match line.`
      : `The walks are ${apart} apart, inside the match line.`
  }
  return `The nearest walk, ${best.personId}, is ${apart} away: past the match line at ${v.threshold}.`
}

export default function Verdict() {
  const clue = getClue()
  const [verdict, setVerdict] = useState<VerdictResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [job, setJob] = useState<JobStatus | null>(null)

  // Saving an unknown walker to the Archive
  const [saveOpen, setSaveOpen] = useState(false)
  const [newName, setNewName] = useState('')
  const [saveError, setSaveError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [savedAs, setSavedAs] = useState<string | null>(null)
  const trimmedName = newName.trim()

  useEffect(() => {
    if (!clue) return
    let live = true
    const { result, stop } = investigateClue((s) => live && setJob(s))
    result
      .then((v) => live && setVerdict(v))
      .catch((e: unknown) => live && setError(e instanceof Error ? e.message : 'Something went wrong.'))
    return () => {
      live = false
      stop()
    }
  }, [clue])

  // Reloaded the page or came here directly: there's no clip to investigate.
  if (!clue) return <Navigate to="/crime-scene" replace />

  const openSaveDialog = () => {
    if (!verdict) return
    setNewName(suggestedName(verdict.gallerySize))
    setSaveError(null)
    setSaveOpen(true)
  }

  const handleSave = async () => {
    if (!verdict || trimmedName === '' || saving) return
    setSaving(true)
    setSaveError(null)
    try {
      const saved = await api.saveQuery(trimmedName, verdict.queryId)
      setSavedAs(saved.personId)
      setSaveOpen(false)
    } catch (e) {
      setSaveError(e instanceof ApiError ? e.message : 'Saving failed.')
    } finally {
      setSaving(false)
    }
  }

  const best = verdict?.suspects[0]
  const clipName = verdict?.clipName ?? clue.name
  const matched = verdict?.isMatch === true && best !== undefined

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
        <h1 className="verdict-heading">{loading ? 'Investigating…' : error ? 'Case on hold' : 'The Verdict'}</h1>
        <p className="verdict-caseline">
          {verdict && <span className="verdict-tag">CASE #{verdict.caseId}</span>}
          evidence: {clipName}
          {verdict && <span className="verdict-tag">compared against {verdict.gallerySize} case files</span>}
        </p>
      </div>

      {loading ? (
        <InvestigationLoader
          done={verdict !== null || error !== null}
          stage={job?.status ?? null}
          progress={job?.progress ?? null}
          onDone={() => setLoading(false)}
        />
      ) : error || !verdict ? (
        <div className="verdict-stage">
          <article className="verdict-bubble verdict-error">
            <span className="verdict-bubble-tab">Case on hold</span>
            <h2 className="verdict-suspect-id">The trail went cold</h2>
            <p className="verdict-suspect-sub">{error}</p>
            <div className="verdict-actions">
              <Button
                component={Link}
                to="/crime-scene"
                variant="contained"
                startIcon={<SearchOutlinedIcon />}
                sx={{ borderRadius: '999px', px: 3, py: 1.25 }}
              >
                Back to the Crime Scene
              </Button>
            </div>
          </article>
          <img src={duckNoGem} className="verdict-duck verdict-duck-nogem" alt="" />
        </div>
      ) : (
        <div className={matched ? 'verdict-stage is-match' : 'verdict-stage is-unknown'}>
          <div className="verdict-case">
            <article className="verdict-bubble">
              <span className="verdict-bubble-tab">{matched ? 'Prime suspect' : 'No match'}</span>

              <div className="verdict-suspect-head">
                <div>
                  <p className="verdict-kicker">{matched ? 'Identified as' : 'The walker is'}</p>
                  <h2 className="verdict-suspect-id">{matched ? best.personId : 'Unknown'}</h2>
                </div>
                <span className={matched ? 'verdict-stamp' : 'verdict-stamp verdict-stamp-unknown'}>
                  {matched ? 'Match found' : savedAs ? 'Case opened' : 'Unidentified'}
                </span>
              </div>

              <p className="verdict-sentence">{verdictSentence(verdict, best)}</p>

              {best && <DistanceScale distance={best.distance} threshold={verdict.threshold} isMatch={matched} />}

              {best && (
                <dl className="verdict-facts">
                  {verdictFacts(verdict, best, matched).map(([label, value]) => (
                    <div key={label}>
                      <dt>{label}</dt>
                      <dd title={value}>{value}</dd>
                    </div>
                  ))}
                </dl>
              )}

              <p className="verdict-fine">
                Distance compares the two 4096-number gait signatures: the smaller, the more alike.
                {!verdict.thresholdCalibrated && ' The match line is provisional until it is calibrated on recorded walks.'}
              </p>
            </article>

            <section className="verdict-evidence-panel" aria-label="The evidence">
              <h2 className="verdict-panel-title">The evidence</h2>
              <div className="verdict-evidence-pair">
                <WalkPlayer frames={verdict.querySilhouettes} label="The clue" title={verdict.clipName} />
                {matched ? (
                  <WalkPlayer frames={best.silhouetteUrls} label="On file" title={best.personId} />
                ) : savedAs ? (
                  <WalkPlayer frames={verdict.querySilhouettes} label="New case file" title={savedAs} />
                ) : (
                  <EmptyCaseFile>
                    <p>
                      {verdict.gallerySize === 0
                        ? 'The Archive is empty. Open the first case file so Swalker recognises this walker next time.'
                        : 'Nobody in the Archive walks like this. Open a case file so Swalker recognises them next time.'}
                    </p>
                    <Button
                      variant="contained"
                      startIcon={<ArchiveOutlinedIcon />}
                      onClick={openSaveDialog}
                      sx={{ borderRadius: '999px', px: 2.5 }}
                    >
                      Open a case file
                    </Button>
                  </EmptyCaseFile>
                )}
              </div>
              {savedAs && (
                <p className="verdict-saved">
                  <CheckCircleOutlinedIcon fontSize="small" />
                  Saved to the Archive as <b>{savedAs}</b>
                  <Link to={`/archive?person=${encodeURIComponent(savedAs)}`}>See the case file →</Link>
                </p>
              )}
            </section>
          </div>

          <div className="verdict-lower">
            {verdict.suspects.length > 0 && (
              <section className="verdict-lineup-panel" aria-label={matched ? 'The lineup' : 'Nearest walks'}>
                <h2 className="verdict-side-title">
                  {matched ? 'The Lineup' : 'Nearest walks'}
                  <span>{matched ? 'closest walks, ranked by distance' : 'none of these crossed the match line'}</span>
                </h2>
                <div className="verdict-lineup">
                  {verdict.suspects.map((s, i) => (
                    <SuspectCard key={s.personId} suspect={s} rank={i + 1} threshold={verdict.threshold} isPrime={matched && i === 0} />
                  ))}
                </div>
              </section>
            )}

            <section className="verdict-trail-panel" aria-label="How we got here">
              <h2 className="verdict-side-title">How we got here</h2>
              <ol className="verdict-trail">
                <li>
                  <b>1 person</b> followed through {verdict.framesWithPerson} of {verdict.framesTotal} frames
                </li>
                <li>
                  <b>{verdict.framesUsed} silhouettes</b> read in order
                </li>
                <li>
                  <b>{FEATURE_DIM}-number</b> gait signature built
                </li>
                <li>
                  <b>{verdict.gallerySize} case files</b> compared
                </li>
              </ol>
            </section>
          </div>

          {matched ? (
            <img src={duckGem} className="verdict-duck verdict-duck-gem" alt="" />
          ) : (
            <img src={duckNoGem} className="verdict-duck verdict-duck-nogem" alt="" />
          )}
        </div>
      )}

      <Dialog
        open={saveOpen}
        onClose={() => !saving && setSaveOpen(false)}
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
            void handleSave()
          }}
        >
          <DialogTitle sx={{ fontWeight: 700 }}>Open a case file</DialogTitle>
          <DialogContent>
            <p className="verdict-dialog-text">
              Their gait signature from {clipName} is saved under this name. The video itself is not kept.
            </p>
            <TextField
              autoFocus
              fullWidth
              label="Case file name"
              value={newName}
              onChange={(e) => {
                setNewName(e.target.value)
                setSaveError(null)
              }}
              error={saveError !== null}
              helperText={saveError ?? ' '}
              slotProps={{ htmlInput: { maxLength: 40 } }}
              sx={{ mt: 1 }}
            />
          </DialogContent>
          <DialogActions sx={{ px: 3, pb: 2.5 }}>
            <Button onClick={() => setSaveOpen(false)} disabled={saving} sx={{ color: 'var(--muted)' }}>
              Cancel
            </Button>
            <Button
              type="submit"
              variant="contained"
              disabled={trimmedName === '' || saving}
              sx={{ borderRadius: '999px', px: 3 }}
            >
              {saving ? 'Saving…' : 'Save'}
            </Button>
          </DialogActions>
        </form>
      </Dialog>
    </section>
  )
}
