import { useState } from 'react'
import Button from '@mui/material/Button'
import UploadFileOutlinedIcon from '@mui/icons-material/UploadFileOutlined'
import CheckCircleOutlinedIcon from '@mui/icons-material/CheckCircleOutlined'
import SearchOutlinedIcon from '@mui/icons-material/SearchOutlined'
import duckLooking from '../assets/duck_looking.png'

export default function Demo() {
  const [clueAdded, setClueAdded] = useState(false)

  return (
    <section className="page page-demo-content">
      <div className="demo-header">
        <h1 className="demo-heading">The Crime Scene</h1>
      </div>

      <div className="demo-evidence-group">
        <Button
          variant={clueAdded ? 'contained' : 'outlined'}
          startIcon={clueAdded ? <CheckCircleOutlinedIcon /> : <UploadFileOutlinedIcon />}
          onClick={() => setClueAdded(true)}
          sx={{ borderRadius: '999px', px: 3, py: 1 }}
        >
          {clueAdded ? 'Clue Added' : 'Add Clue'}
        </Button>
        <p className="demo-board-text">
          {clueAdded
            ? 'clip_001.mp4 is in — ready for analysis.'
            : 'No clue yet. Add a clip above to open the case.'}
        </p>
      </div>

      <div className="demo-bottom">
        <Button
          variant="contained"
          disabled={!clueAdded}
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
