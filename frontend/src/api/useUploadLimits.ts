import { useEffect, useState } from 'react'
import { api, DEFAULT_LIMITS } from './client'

// The server's upload limits, so files can be checked before a long upload. Falls back to
// the defaults if the server can't be reached (the upload itself will then say so).
export function useUploadLimits() {
  const [limits, setLimits] = useState(DEFAULT_LIMITS)

  useEffect(() => {
    let live = true
    api
      .health()
      .then((h) => live && setLimits({ maxUploadMb: h.maxUploadMb, videoExtensions: h.videoExtensions }))
      .catch(() => {})
    return () => {
      live = false
    }
  }, [])

  return limits
}
