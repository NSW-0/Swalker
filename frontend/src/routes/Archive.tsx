import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import Dialog from '@mui/material/Dialog'
import DialogTitle from '@mui/material/DialogTitle'
import DialogContent from '@mui/material/DialogContent'
import IconButton from '@mui/material/IconButton'
import CloseIcon from '@mui/icons-material/Close'
import AddOutlinedIcon from '@mui/icons-material/AddOutlined'
import { FEATURE_DIM } from '../data/mockGallery'
import type { GalleryEntry } from '../data/mockGallery'
import { getArchive } from '../data/archiveStore'
import duckSmoking from '../assets/duck_smoking.png'

function l2Norm(vector: number[]) {
  return Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0))
}

export default function Archive() {
  const [entries] = useState(getArchive)
  // /archive?person=person_021 opens that case file straight away
  // (used after saving an unknown walker on the Verdict page).
  const [searchParams] = useSearchParams()
  const [selected, setSelected] = useState<GalleryEntry | null>(
    () => entries.find((e) => e.personId === searchParams.get('person')) ?? null,
  )

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
        <div className="archive-grid">
          <button type="button" className="archive-card archive-card-add">
            <AddOutlinedIcon className="archive-card-add-icon" />
            <p className="archive-card-id">Add Case File</p>
          </button>

          {entries.map((entry) => (
            <button
              key={entry.personId}
              type="button"
              className="archive-card"
              onClick={() => setSelected(entry)}
            >
              <p className="archive-card-id">{entry.personId}</p>
              <p className="archive-card-hint">View case file →</p>
            </button>
          ))}
        </div>
      </div>

      <Dialog
        open={selected !== null}
        onClose={() => setSelected(null)}
        maxWidth="sm"
        fullWidth
        slotProps={{
          paper: {
            sx: {
              background: 'rgba(54, 53, 58, 0.97)',
              backgroundImage: 'none',
              border: '1px solid var(--border)',
              borderRadius: '12px',
              backdropFilter: 'blur(12px)',
            },
          },
        }}
      >
        {selected && (
          <>
            <DialogTitle className="archive-dialog-title">
              {selected.personId}
              <IconButton
                onClick={() => setSelected(null)}
                sx={{ position: 'absolute', right: 12, top: 12, color: 'var(--muted)' }}
              >
                <CloseIcon />
              </IconButton>
            </DialogTitle>
            <DialogContent>
              <p className="archive-detail-row">
                <span className="archive-detail-label">Video:</span>
                {selected.videoPath}
              </p>

              <div className="archive-vector-box">
                <span className="archive-badge">{FEATURE_DIM}-dim vector</span>
                <span className="archive-badge">L2 norm {l2Norm(selected.featureVector).toFixed(3)}</span>
                <p className="archive-vector-preview">
                  [{selected.featureVector.slice(0, 8).map((v) => v.toFixed(4)).join(', ')}, …]
                </p>
              </div>
            </DialogContent>
          </>
        )}
      </Dialog>

      <img src={duckSmoking} className="archive-duck" alt="" />
    </section>
  )
}
