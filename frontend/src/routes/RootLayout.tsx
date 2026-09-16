import { Outlet } from 'react-router-dom'
import Navbar from '../components/layout/Navbar'

export default function RootLayout() {
  return (
    <div className="app-shell">
      <Navbar />
      <main className="app-content">
        <Outlet />
      </main>
    </div>
  )
}
