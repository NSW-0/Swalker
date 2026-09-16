import { NavLink } from 'react-router-dom'
import HomeOutlinedIcon from '@mui/icons-material/HomeOutlined'
import PlayCircleOutlinedIcon from '@mui/icons-material/PlayCircleOutlined'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'

const links = [
  { to: '/', label: 'Home', end: true, Icon: HomeOutlinedIcon },
  { to: '/demo', label: 'Demo', Icon: PlayCircleOutlinedIcon },
  { to: '/about', label: 'About', Icon: InfoOutlinedIcon },
]

export default function Navbar() {
  return (
    <header className="navbar">
      <div className="navbar-inner">
        <nav>
          {links.map(({ to, label, end, Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) => (isActive ? 'nav-link active' : 'nav-link')}
            >
              <Icon fontSize="inherit" />
              {label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}
