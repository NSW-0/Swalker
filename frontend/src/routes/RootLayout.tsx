import { Outlet, useLocation } from 'react-router-dom'
import Navbar from '../components/layout/Navbar'

const pageBackgroundClass: Record<string, string> = {
  '/crime-scene': 'page-demo',
  '/archive': 'page-archive',
}

export default function RootLayout() {
  const location = useLocation()
  const backgroundClass = pageBackgroundClass[location.pathname] ?? ''

  return (
    <div className={`app-shell ${backgroundClass}`.trim()}>
      <Navbar />
      <main className="app-content">
        <Outlet />
      </main>
    </div>
  )
}
