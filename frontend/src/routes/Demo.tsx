import { useState } from 'react'
import Button from '@mui/material/Button'
import UploadFileOutlinedIcon from '@mui/icons-material/UploadFileOutlined'
import CheckCircleOutlinedIcon from '@mui/icons-material/CheckCircleOutlined'
import SearchOutlinedIcon from '@mui/icons-material/SearchOutlined'
import duckLooking from '../assets/duck_looking.png'

export default function Demo() {
  const [evidenceAdded, setEvidenceAdded] = useState(false)

  return (
    <section className="page page-demo-content">
      <div className="demo-header">
        <h1 className="demo-heading">The Evidence Room</h1>
      </div>

      <div className="demo-evidence-group">
        <Button
          variant={evidenceAdded ? 'contained' : 'outlined'}
          startIcon={evidenceAdded ? <CheckCircleOutlinedIcon /> : <UploadFileOutlinedIcon />}
          onClick={() => setEvidenceAdded(true)}
          sx={{ borderRadius: '999px', px: 3, py: 1 }}
        >
          {evidenceAdded ? 'Evidence Added' : 'Add Evidence'}
        </Button>
        <p className="demo-board-text">
          {evidenceAdded
            ? 'clip_001.mp4 is in — ready for analysis.'
            : 'No evidence yet. Add a clip above to open the case.'}
        </p>
      </div>

      <div className="demo-bottom">
        <Button
          variant="contained"
          disabled={!evidenceAdded}
          startIcon={<SearchOutlinedIcon />}
          sx={{ borderRadius: '999px', px: 4, py: 1.5, fontSize: '1.05rem' }}
        >
          Investigate the Evidence
        </Button>
      </div>

      <img src={duckLooking} className="demo-duck" alt="" />
    </section>
  )
}
