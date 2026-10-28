// The clip picked on the Crime Scene, handed to the Verdict page. Memory only: after a reload
// there is no clue and the Verdict page sends the visitor back to pick one.
//
// The identify request is kept per clue, so React StrictMode's double effect (and coming back
// to the page) reuses one request instead of making the server analyse the same video twice.

import { api } from '../api/client'
import type { VerdictResult } from '../api/client'

let clue: File | null = null
let verdict: Promise<VerdictResult> | null = null

export function setClue(file: File) {
  clue = file
  verdict = null
}

export function getClue() {
  return clue
}

export function investigateClue(): Promise<VerdictResult> {
  if (!clue) return Promise.reject(new Error('No clue picked'))
  verdict ??= api.identify(clue).catch((err: unknown) => {
    verdict = null // a failed attempt can be retried
    throw err
  })
  return verdict
}
