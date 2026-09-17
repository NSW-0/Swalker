import { createBrowserRouter } from 'react-router-dom'
import RootLayout from './routes/RootLayout'
import Home from './routes/Home'
import Demo from './routes/Demo'
import Archive from './routes/Archive'
import About from './routes/About'
import NotFound from './routes/NotFound'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <RootLayout />,
    children: [
      { index: true, element: <Home /> },
      { path: 'demo', element: <Demo /> },
      { path: 'archive', element: <Archive /> },
      { path: 'about', element: <About /> },
      { path: '*', element: <NotFound /> },
    ],
  },
])
