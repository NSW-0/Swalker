import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider } from 'react-router-dom'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import { router } from './router'
import './index.css'
import './i18n'

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#F58F7C' },
    secondary: { main: '#F2C4CE' },
    background: {
      default: '#2C2B30',
      paper: '#36353A',
    },
  },
  typography: {
    fontFamily: "system-ui, 'Segoe UI', Roboto, sans-serif",
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <RouterProvider router={router} />
    </ThemeProvider>
  </StrictMode>,
)
