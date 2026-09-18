import { Link } from 'react-router-dom'
import Button from '@mui/material/Button'
import duck from '../assets/duck.png'

export default function Home() {
  return (
    <section className="page page-hero">
      <h1>
        Someone's walking by,
        <br />
        <span className="hero-accent">you have no idea who it is.</span>
      </h1>
      <p className="hero-sub">
        <span className="hero-brand">
          <span className="hero-brand-accent">S</span>
          <span className="hero-brand-white">walker</span>
        </span>{' '}
        identifies people by the unique way they walk — no face needed.
      </p>
      <Button
        component={Link}
        to="/crime-scene"
        variant="contained"
        sx={{ borderRadius: '999px', px: 4, py: 1.5, fontSize: '1.05rem' }}
      >
        Let's Investigate
      </Button>
      <img src={duck} className="hero-duck" alt="" />
    </section>
  )
}
