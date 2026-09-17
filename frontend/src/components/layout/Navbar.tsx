import { NavLink, useLocation } from 'react-router-dom'
import Tooltip from '@mui/material/Tooltip'
import HomeOutlinedIcon from '@mui/icons-material/HomeOutlined'
import PlayCircleOutlinedIcon from '@mui/icons-material/PlayCircleOutlined'
import Inventory2OutlinedIcon from '@mui/icons-material/Inventory2Outlined'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'

const links = [
  { to: '/', label: 'Home', end: true, Icon: HomeOutlinedIcon },
  { to: '/demo', label: 'Demo', Icon: PlayCircleOutlinedIcon },
  { to: '/archive', label: 'Archive', Icon: Inventory2OutlinedIcon },
  { to: '/about', label: 'About', Icon: InfoOutlinedIcon },
]

export default function Navbar() {
  const location = useLocation()

  return (
    <header className="navbar">
      <div className="navbar-inner">
        <nav>
          {links.map(({ to, label, end, Icon }) => {
            const isActive = end ? location.pathname === to : location.pathname.startsWith(to)

            return (
              <Tooltip
                key={to}
                title={label}
                placement="right"
                arrow
                slotProps={{
                  tooltip: {
                    sx: {
                      bgcolor: 'rgba(54, 53, 58, 0.97)',
                      border: '1px solid var(--border)',
                      fontSize: '13px',
                      fontWeight: 600,
                    },
                  },
                  arrow: {
                    sx: { color: 'rgba(54, 53, 58, 0.97)' },
                  },
                }}
              >
                <NavLink to={to} end={end} className={isActive ? 'nav-link active' : 'nav-link'}>
                  <Icon fontSize="inherit" />
                  <span className="nav-label">{label}</span>
                </NavLink>
              </Tooltip>
            )
          })}
        </nav>
      </div>
    </header>
  )
}
