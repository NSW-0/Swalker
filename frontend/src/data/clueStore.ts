// The clip picked on the Crime Scene, handed to the Verdict page. Memory only: after a reload
// there is no clue and the Verdict page sends the visitor back to pick one.
//
// Investigating starts one identify job per clue and polls its status. The run is kept per clue,
// so React StrictMode's double effect (and coming back to the page) reuses one job instead of
// making the server analyse the same video twice.

import { api, ApiError } from '../api/client'
import type { JobStatus, VerdictResult } from '../api/client'

const POLL_MS = 400

interface Run {
  result: Promise<VerdictResult>
  status: JobStatus | null
  listeners: Set<(s: JobStatus) => void>
}

let clue: File | null = null
let run: Run | null = null

export function setClue(file: File) {
  clue = file
  run = null
}

export function getClue() {
  return clue
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

function start(file: File): Run {
  const r: Run = { result: Promise.resolve(null as never), status: null, listeners: new Set() }
  r.result = (async () => {
    const { jobId } = await api.startIdentify(file)
    for (;;) {
      const status = await api.identifyStatus(jobId)
      r.status = status
      r.listeners.forEach((listener) => listener(status))
      if (status.status === 'done' && status.result) return status.result
      if (status.status === 'error') {
        throw new ApiError(status.error?.error ?? 'server_error', status.error?.message ?? 'Investigation failed', 0)
      }
      await sleep(POLL_MS)
    }
  })().catch((err: unknown) => {
    if (run === r) run = null // a failed attempt can be retried
    throw err
  })
  return r
}

// Starts (or rejoins) the investigation of the current clue. onStatus hears every status poll;
// call stop() to stop listening (the job itself keeps going on the server).
export function investigateClue(onStatus?: (s: JobStatus) => void): { result: Promise<VerdictResult>; stop: () => void } {
  if (!clue) return { result: Promise.reject(new Error('No clue picked')), stop: () => {} }
  run ??= start(clue)
  const r = run
  if (onStatus) {
    r.listeners.add(onStatus)
    if (r.status) onStatus(r.status)
  }
  return { result: r.result, stop: () => onStatus && r.listeners.delete(onStatus) }
}
