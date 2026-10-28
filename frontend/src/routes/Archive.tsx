import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import Button from '@mui/material/Button'
import CircularProgress from '@mui/material/CircularProgress'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogTitle from '@mui/material/DialogTitle'
import DialogContent from '@mui/material/DialogContent'
import IconButton from '@mui/material/IconButton'
import TextField from '@mui/material/TextField'
import CloseIcon from '@mui/icons-material/Close'
import AddOutlinedIcon from '@mui/icons-material/AddOutlined'
import UploadFileOutlinedIcon from '@mui/icons-material/UploadFileOutlined'
import EditOutlinedIcon from '@mui/icons-material/EditOutlined'
import DeleteOutlinedIcon from '@mui/icons-material/DeleteOutlined'
import { api, ApiError, checkVideo, formatSize } from '../api/client'
import type { CaseFile, CaseFileDetail } from '../api/client'
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

function AddCaseFileDialog({ open, onClose, onAdded }: { open: boolean; onClose: () => void; onAdded: (c: CaseFileDetail) => void }) {
  const limits = useUploadLimits()
  const inputRef = useRef<HTMLInputElement>(null)
  const [name, setName] = useState('')
  const [clip, setClip] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const reset = () => {
    setName('')
    setClip(null)
    setError(null)
  }

  const submit = async () => {
    if (!clip || name.trim() === '' || busy) return
    setBusy(true)
    setError(null)
    try {
      const added = await api.addFromVideo(name.trim(), clip)
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
        <DialogTitle sx={{ fontWeight: 700 }}>Add Case File</DialogTitle>
        <DialogContent>
          <p className="archive-dialog-text">
            Upload a walking clip of someone new. Swalker keeps their gait signature and a few silhouettes; the
            video itself is not stored.
          </p>
          <TextField
            autoFocus
            fullWidth
            label="Case file name"
            value={name}
            onChange={(e) => {
              setName(e.target.value)
              setError(null)
            }}
            slotProps={{ htmlInput: { maxLength: 40 } }}
            sx={{ mt: 1, mb: 2 }}
          />
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
          <Button type="submit" variant="contained" disabled={!clip || name.trim() === '' || busy} sx={{ borderRadius: '999px', px: 3 }}>
            {busy ? 'Adding…' : 'Add'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  )
}

// Rename / delete at the bottom of an open case file. Delete asks first, inside the dialog.
function CaseFileActions({
  personId,
  onRenamed,
  onDeleted,
}: {
  personId: string
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
          Delete <b>{personId}</b>? Their gait signature and silhouettes are removed for good, and Swalker won't
          recognise them any more.
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
          <button type="button" className="archive-card archive-card-add" onClick={() => setAddOpen(true)}>
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
                <p className="archive-card-hint">View case file →</p>
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
                <span className="archive-detail-label">Video:</span>
                {shown.videoPath} <span className="archive-detail-note">(not stored)</span>
              </p>
              <p className="archive-detail-row">
                <span className="archive-detail-label">Added:</span>
                {new Date(shown.addedAt).toLocaleString()}
              </p>

              <SilhouetteStrip frames={shown.silhouetteUrls} className="archive-sil-strip" />

              <div className="archive-vector-box">
                <span className="archive-badge">{shown.featureDim}-dim vector</span>
                <span className="archive-badge">L2 norm {shown.l2Norm.toFixed(3)}</span>
                <p className="archive-vector-preview">[{shown.vectorPreview.map((v) => v.toFixed(4)).join(', ')}, …]</p>
              </div>
            </>
          )}
        </DialogContent>
        {shown && (
          <DialogActions sx={{ px: 3, pb: 2.5 }}>
            <CaseFileActions
              key={shown.personId}
              personId={shown.personId}
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
        open={addOpen}
        onClose={() => setAddOpen(false)}
        onAdded={(added) => {
          setAddOpen(false)
          load()
          openCase(added.personId)
        }}
      />

      <img src={duckSmoking} className="archive-duck" alt="" />
    </section>
  )
}
