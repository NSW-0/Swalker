import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Button from '@mui/material/Button'
import UploadFileOutlinedIcon from '@mui/icons-material/UploadFileOutlined'
import CheckCircleOutlinedIcon from '@mui/icons-material/CheckCircleOutlined'
import SearchOutlinedIcon from '@mui/icons-material/SearchOutlined'
import { checkVideo, formatSize } from '../api/client'
import { useUploadLimits } from '../api/useUploadLimits'
import { getClue, setClue } from '../data/clueStore'
import duckLooking from '../assets/duck_looking.png'

export default function Demo() {
  const navigate = useNavigate()
  const limits = useUploadLimits()
  const inputRef = useRef<HTMLInputElement>(null)
  // Coming back from the Verdict keeps the last clue picked.
  const [clip, setClip] = useState<File | null>(getClue)
  const [problem, setProblem] = useState<string | null>(null)

  const handlePick = (file: File | undefined) => {
    if (!file) return
    const issue = checkVideo(file, limits)
    setProblem(issue)
    setClip(issue ? null : file)
  }

  const investigate = () => {
    if (!clip) return
    setClue(clip)
    navigate('/crime-scene/verdict')
  }

  return (
    <section className="page page-demo-content">
      <div className="demo-header">
        <h1 className="demo-heading">The Crime Scene</h1>
      </div>

      <div className="demo-evidence-group">
        <input
          ref={inputRef}
          type="file"
          accept={['video/*', ...limits.videoExtensions].join(',')}
          hidden
          onChange={(e) => {
            handlePick(e.target.files?.[0])
            e.target.value = '' // picking the same file again still fires onChange
          }}
        />
        <Button
          variant={clip ? 'contained' : 'outlined'}
          startIcon={clip ? <CheckCircleOutlinedIcon /> : <UploadFileOutlinedIcon />}
          onClick={() => inputRef.current?.click()}
          sx={{ borderRadius: '999px', px: 3, py: 1 }}
        >
          {clip ? 'Change Clue' : 'Add Clue'}
        </Button>
        <p className="demo-board-text">
          {clip ? (
            <>
              <span className="demo-clip-name">{clip.name}</span> is in ({formatSize(clip.size)}), ready for analysis.
            </>
          ) : (
            'No clue yet. Add a walking clip to open the case.'
          )}
        </p>
        {problem && (
          <p className="demo-board-text demo-board-problem" role="alert">
            {problem}
          </p>
        )}
        <p className="demo-board-text demo-board-hint">
          Side view, whole body in frame, 3 to 5 seconds of walking. Up to {limits.maxUploadMb} MB.
        </p>
      </div>

      <div className="demo-bottom">
        <Button
          variant="contained"
          disabled={!clip}
          onClick={investigate}
          startIcon={<SearchOutlinedIcon />}
          sx={{ borderRadius: '999px', px: 4, py: 1.5, fontSize: '1.05rem' }}
        >
          Investigate the Clue
        </Button>
      </div>

      <img src={duckLooking} className="demo-duck" alt="" />
    </section>
  )
}
