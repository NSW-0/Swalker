import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import Button from '@mui/material/Button'
import CircularProgress from '@mui/material/CircularProgress'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogTitle from '@mui/material/DialogTitle'
import DialogContent from '@mui/material/DialogContent'
import IconButton from '@mui/material/IconButton'
import MenuItem from '@mui/material/MenuItem'
import TextField from '@mui/material/TextField'
import ToggleButton from '@mui/material/ToggleButton'
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup'
import CloseIcon from '@mui/icons-material/Close'
import AddOutlinedIcon from '@mui/icons-material/AddOutlined'
import UploadFileOutlinedIcon from '@mui/icons-material/UploadFileOutlined'
import EditOutlinedIcon from '@mui/icons-material/EditOutlined'
import DeleteOutlinedIcon from '@mui/icons-material/DeleteOutlined'
import { api, ApiError, checkVideo, formatSize } from '../api/client'
import type { CaseFile, CaseFileDetail, ClipInfo } from '../api/client'
import { useUploadLimits } from '../api/useUploadLimits'
import { SilhouetteStrip, SilhouetteThumb } from '../components/verdict/Silhouettes'
import duckSmoking from '../assets/duck_smoking.png'

const dialogPaperSx = {
  background: 'rgba(54, 53, 58, 0.97)',
  backgroundImage: 'none',
  border: '1px solid var(--border)',
  borderRadius: '12px',
  backdropFilter: 'blur(12px)',
}

function errorMessage(e: unknown) {
  return e instanceof Error ? e.message : 'Something went wrong.'
}

// The case file name used when none is typed: the clip's file name without its extension, with
// characters names can't have turned into spaces (same rule as the backend).
function nameFromFile(file: File) {
  const stem = file.name.replace(/\.[^.]*$/, '')
  return stem.replace(/[^\p{L}\p{N}_ .'-]+/gu, ' ').split(/\s+/).join(' ').trim().slice(0, 40).trim()
}

// Adds a new person, or another clip of someone already on file (forPerson preselects them).
function AddCaseFileDialog({
  open,
  people,
  forPerson,
  onClose,
  onAdded,
}: {
  open: boolean
  people: string[]
  forPerson: string | null
  onClose: () => void
  onAdded: (c: CaseFileDetail) => void
}) {
  const limits = useUploadLimits()
  const inputRef = useRef<HTMLInputElement>(null)
  const [mode, setMode] = useState<'new' | 'existing'>(forPerson ? 'existing' : 'new')
  const [person, setPerson] = useState(forPerson ?? '')
  const [name, setName] = useState('')
  const [clip, setClip] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const fallbackName = clip ? nameFromFile(clip) : ''
  const finalName = name.trim() || fallbackName
  const ready = clip !== null && (mode === 'new' ? finalName !== '' : person !== '')

  const reset = () => {
    setName('')
    setClip(null)
    setError(null)
  }

  const submit = async () => {
    if (!clip || !ready || busy) return
    setBusy(true)
    setError(null)
    try {
      const added = mode === 'new' ? await api.addFromVideo(finalName, clip) : await api.addClipFromVideo(person, clip)
      reset()
      onAdded(added)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : errorMessage(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog open={open} onClose={() => !busy && onClose()} maxWidth="xs" fullWidth slotProps={{ paper: { sx: dialogPaperSx } }}>
      <form
        onSubmit={(e) => {
          e.preventDefault()
          void submit()
        }}
      >
        <DialogTitle sx={{ fontWeight: 700 }}>{mode === 'new' ? 'Add Case File' : 'Add a clip'}</DialogTitle>
        <DialogContent>
          <ToggleButtonGroup
            exclusive
            fullWidth
            size="small"
            value={mode}
            onChange={(_, value: 'new' | 'existing' | null) => {
              if (value) setMode(value)
              setError(null)
            }}
            sx={{ mb: 1.5 }}
          >
            <ToggleButton value="new">New person</ToggleButton>
            <ToggleButton value="existing" disabled={people.length === 0}>
              Existing person
            </ToggleButton>
          </ToggleButtonGroup>
          <p className="archive-dialog-text">
            {mode === 'new'
              ? 'Upload a walking clip of someone new.'
              : 'Another clip of someone already on file, e.g. carrying a bag or in a coat. New walks are compared with each of their clips.'}{' '}
            Swalker keeps the gait signature and a few silhouettes; the video itself is not stored.
          </p>
          {mode === 'new' ? (
            <TextField
              autoFocus
              fullWidth
              label="Case file name"
              placeholder={fallbackName || undefined}
              helperText={fallbackName ? `Leave empty to use the clip's name: ${fallbackName}` : 'Optional: the clip\'s name is used if left empty'}
              value={name}
              onChange={(e) => {
                setName(e.target.value)
                setError(null)
              }}
              slotProps={{ htmlInput: { maxLength: 40 } }}
              sx={{ mt: 1, mb: 2 }}
            />
          ) : (
            <TextField
              select
              fullWidth
              label="Person"
              value={person}
              onChange={(e) => {
                setPerson(e.target.value)
                setError(null)
              }}
              sx={{ mt: 1, mb: 2 }}
            >
              {people.map((id) => (
                <MenuItem key={id} value={id}>
                  {id}
                </MenuItem>
              ))}
            </TextField>
          )}
          <input
            ref={inputRef}
            type="file"
            accept={['video/*', ...limits.videoExtensions].join(',')}
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0]
              e.target.value = ''
              if (!file) return
              const issue = checkVideo(file, limits)
              setError(issue)
              setClip(issue ? null : file)
            }}
          />
          <Button
            variant="outlined"
            startIcon={<UploadFileOutlinedIcon />}
            onClick={() => inputRef.current?.click()}
            disabled={busy}
            sx={{ borderRadius: '999px' }}
          >
            {clip ? 'Change clip' : 'Choose walking clip'}
          </Button>
          {clip && (
            <p className="archive-dialog-text archive-dialog-file">
              {clip.name} · {formatSize(clip.size)}
            </p>
          )}
          {busy && (
            <p className="archive-dialog-text archive-dialog-busy">
              <CircularProgress size={16} /> Reading the walk… this can take a while on CPU.
            </p>
          )}
          {error && (
            <p className="archive-dialog-text archive-dialog-error" role="alert">
              {error}
            </p>
          )}
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2.5 }}>
          <Button onClick={onClose} disabled={busy} sx={{ color: 'var(--muted)' }}>
            Cancel
          </Button>
          <Button type="submit" variant="contained" disabled={!ready || busy} sx={{ borderRadius: '999px', px: 3 }}>
            {busy ? 'Adding…' : 'Add'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  )
}

// Each clip of a person: its silhouettes and signature, and a remove button (asks first) when
// they have more than one.
function ClipList({ personId, clips, onChanged }: { personId: string; clips: ClipInfo[]; onChanged: (c: CaseFileDetail) => void }) {
  const [confirming, setConfirming] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const remove = async (clipId: string) => {
    setBusy(true)
    setError(null)
    try {
      onChanged(await api.removeClip(personId, clipId))
      setConfirming(null)
    } catch (e) {
      setError(errorMessage(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="archive-clips">
      {clips.map((clip, i) => (
        <div key={clip.clipId} className="archive-clip">
          <div className="archive-clip-head">
            <span className="archive-clip-title">
              Clip {i + 1} · <b>{clip.videoPath}</b> <span className="archive-detail-note">(not stored)</span>
            </span>
            <span className="archive-clip-date">{new Date(clip.addedAt).toLocaleString()}</span>
            {clips.length > 1 &&
              (confirming === clip.clipId ? (
                <span className="archive-clip-confirm">
                  Remove this clip?
                  <Button size="small" onClick={() => setConfirming(null)} disabled={busy} sx={{ color: 'var(--muted)' }}>
                    Keep
                  </Button>
                  <Button size="small" color="error" onClick={() => void remove(clip.clipId)} disabled={busy}>
                    {busy ? 'Removing…' : 'Remove'}
                  </Button>
                </span>
              ) : (
                <Button
                  size="small"
                  startIcon={<DeleteOutlinedIcon />}
                  onClick={() => {
                    setConfirming(clip.clipId)
                    setError(null)
                  }}
                  sx={{ color: 'var(--muted)' }}
                >
                  Remove
                </Button>
              ))}
          </div>
          <SilhouetteStrip frames={clip.silhouetteUrls} className="archive-sil-strip" />
          <div className="archive-vector-box">
            <span className="archive-badge">{clip.featureDim}-dim vector</span>
            <span className="archive-badge">L2 norm {clip.l2Norm.toFixed(3)}</span>
            <p className="archive-vector-preview">[{clip.vectorPreview.map((v) => v.toFixed(4)).join(', ')}, …]</p>
          </div>
        </div>
      ))}
      {error && <p className="archive-dialog-text archive-dialog-error">{error}</p>}
    </div>
  )
}

// Rename / delete at the bottom of an open case file. Delete asks first, inside the dialog.
function CaseFileActions({
  personId,
  onAddClip,
  onRenamed,
  onDeleted,
}: {
  personId: string
  onAddClip: () => void
  onRenamed: (c: CaseFileDetail) => void
  onDeleted: () => void
}) {
  const [mode, setMode] = useState<'idle' | 'rename' | 'delete'>('idle')
  const [name, setName] = useState(personId)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const run = async (action: () => Promise<void>) => {
    setBusy(true)
    setError(null)
    try {
      await action()
    } catch (e) {
      setError(errorMessage(e))
    } finally {
      setBusy(false)
    }
  }

  const rename = () =>
    run(async () => {
      const trimmed = name.trim()
      if (trimmed === personId) return setMode('idle')
      onRenamed(await api.renameCaseFile(personId, trimmed))
    })

  const remove = () =>
    run(async () => {
      await api.deleteCaseFile(personId)
      onDeleted()
    })

  if (mode === 'rename') {
    return (
      <form
        className="archive-actions archive-actions-rename"
        onSubmit={(e) => {
          e.preventDefault()
          void rename()
        }}
      >
        <TextField
          autoFocus
          size="small"
          label="New name"
          value={name}
          onChange={(e) => {
            setName(e.target.value)
            setError(null)
          }}
          error={error !== null}
          helperText={error ?? ' '}
          slotProps={{ htmlInput: { maxLength: 40 } }}
          sx={{ flex: 1 }}
        />
        <Button onClick={() => setMode('idle')} disabled={busy} sx={{ color: 'var(--muted)' }}>
          Cancel
        </Button>
        <Button type="submit" variant="contained" disabled={busy || name.trim() === ''} sx={{ borderRadius: '999px' }}>
          {busy ? 'Saving…' : 'Save'}
        </Button>
      </form>
    )
  }

  if (mode === 'delete') {
    return (
      <div className="archive-actions archive-actions-delete" role="alertdialog" aria-label="Confirm delete">
        <p className="archive-dialog-text">
          Delete <b>{personId}</b>? All their clips' gait signatures and silhouettes are removed for good, and
          Swalker won't recognise them any more.
        </p>
        {error && <p className="archive-dialog-text archive-dialog-error">{error}</p>}
        <div className="archive-actions-row">
          <Button onClick={() => setMode('idle')} disabled={busy} sx={{ color: 'var(--muted)' }}>
            Keep it
          </Button>
          <Button variant="contained" color="error" onClick={() => void remove()} disabled={busy} sx={{ borderRadius: '999px' }}>
            {busy ? 'Deleting…' : 'Delete for good'}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="archive-actions archive-actions-row">
      <Button startIcon={<AddOutlinedIcon />} onClick={onAddClip} sx={{ color: 'var(--muted)', mr: 'auto' }}>
        Add clip
      </Button>
      <Button startIcon={<EditOutlinedIcon />} onClick={() => setMode('rename')} sx={{ color: 'var(--muted)' }}>
        Rename
      </Button>
      <Button startIcon={<DeleteOutlinedIcon />} onClick={() => setMode('delete')} sx={{ color: 'var(--accent)' }}>
        Delete
      </Button>
    </div>
  )
}

export default function Archive() {
  const [entries, setEntries] = useState<CaseFile[] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [addOpen, setAddOpen] = useState(false)
  const [addFor, setAddFor] = useState<string | null>(null) // add a clip to this person

  // /archive?person=person_021 opens that case file straight away
  // (used after saving an unknown walker on the Verdict page).
  const [searchParams, setSearchParams] = useSearchParams()
  const selectedId = searchParams.get('person')
  const [detail, setDetail] = useState<CaseFileDetail | null>(null)
  const [detailError, setDetailError] = useState<string | null>(null)
  // After back/forward the previous person's details may still be in state; don't show them.
  const shown = detail?.personId === selectedId ? detail : null

  const load = useCallback(() => {
    api
      .gallery()
      .then((g) => {
        setEntries(g.entries)
        setLoadError(null)
      })
      .catch((e: unknown) => setLoadError(errorMessage(e)))
  }, [])

  useEffect(load, [load])

  useEffect(() => {
    if (!selectedId) return
    let live = true
    api
      .caseFile(selectedId)
      .then((d) => live && setDetail(d))
      .catch((e: unknown) => live && setDetailError(errorMessage(e)))
    return () => {
      live = false
    }
  }, [selectedId])

  const openCase = (personId: string) => {
    setDetail(null)
    setDetailError(null)
    setSearchParams({ person: personId })
  }
  const closeCase = () => setSearchParams({})

  return (
    <section className="page page-archive-content">
      <div className="archive-header">
        <h1 className="archive-heading">The Archive</h1>
      </div>
      <p className="archive-subtext">
        Every walk{' '}
        <span className="hero-brand">
          <span className="hero-brand-accent">S</span>
          <span className="hero-brand-white">walker</span>
        </span>{' '}
        has learned to recognize.
        <br />
        Open a case file to see what the model stored for that person.
      </p>

      <div className="archive-grid-scroll">
        {loadError && (
          <p className="archive-status" role="alert">
            {loadError}{' '}
            <Button size="small" onClick={load}>
              Try again
            </Button>
          </p>
        )}
        {!entries && !loadError && <p className="archive-status">Opening the filing cabinet…</p>}
        {entries && entries.length === 0 && (
          <p className="archive-status">The Archive is empty. Add the first case file to start recognising walkers.</p>
        )}
        <div className="archive-grid">
          <button
            type="button"
            className="archive-card archive-card-add"
            onClick={() => {
              setAddFor(null)
              setAddOpen(true)
            }}
          >
            <AddOutlinedIcon className="archive-card-add-icon" />
            <p className="archive-card-id">Add Case File</p>
          </button>

          {entries?.map((entry) => (
            <button key={entry.personId} type="button" className="archive-card archive-card-person" onClick={() => openCase(entry.personId)}>
              <span className="archive-card-thumb">
                <SilhouetteThumb src={entry.thumbnailUrl} />
              </span>
              <div>
                <p className="archive-card-id">{entry.personId}</p>
                <p className="archive-card-hint">
                  {entry.clipCount > 1 ? `${entry.clipCount} clips · ` : ''}View case file →
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>

      <Dialog open={selectedId !== null} onClose={closeCase} maxWidth="sm" fullWidth slotProps={{ paper: { sx: dialogPaperSx } }}>
        <DialogTitle className="archive-dialog-title">
          {selectedId}
          <IconButton onClick={closeCase} aria-label="Close" sx={{ position: 'absolute', right: 12, top: 12, color: 'var(--muted)' }}>
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          {detailError && <p className="archive-dialog-text archive-dialog-error">{detailError}</p>}
          {!shown && !detailError && <CircularProgress size={24} />}
          {shown && (
            <>
              <p className="archive-detail-row">
                <span className="archive-detail-label">Added:</span>
                {new Date(shown.addedAt).toLocaleString()}
              </p>
              <p className="archive-detail-row">
                <span className="archive-detail-label">Clips:</span>
                {shown.clipCount}
                <span className="archive-detail-note">
                  {' '}
                  (a new walk is compared with each; add clips with a bag or a coat to recognise those too)
                </span>
              </p>

              <ClipList
                personId={shown.personId}
                clips={shown.clips}
                onChanged={(changed) => {
                  setDetail(changed)
                  load()
                }}
              />
            </>
          )}
        </DialogContent>
        {shown && (
          <DialogActions sx={{ px: 3, pb: 2.5 }}>
            <CaseFileActions
              key={shown.personId}
              personId={shown.personId}
              onAddClip={() => {
                setAddFor(shown.personId)
                setAddOpen(true)
              }}
              onRenamed={(renamed) => {
                setDetail(renamed)
                setSearchParams({ person: renamed.personId }, { replace: true })
                load()
              }}
              onDeleted={() => {
                closeCase()
                load()
              }}
            />
          </DialogActions>
        )}
      </Dialog>

      <AddCaseFileDialog
        key={addFor ?? '(new)'}
        open={addOpen}
        people={entries?.map((e) => e.personId) ?? []}
        forPerson={addFor}
        onClose={() => setAddOpen(false)}
        onAdded={(added) => {
          setAddOpen(false)
          load()
          if (added.personId === selectedId) setDetail(added)
          else openCase(added.personId)
        }}
      />

      <img src={duckSmoking} className="archive-duck" alt="" />
    </section>
  )
}
