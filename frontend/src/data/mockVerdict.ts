// Mock stand-in for what identify.py returns, until the backend exists.
// It uses the same math as identify_from_features(): Euclidean distance
// against every gallery vector, keep the top K, and turn distances into a
// "confidence" that is only relative between those K (they add up to 100).
// A real API response can replace buildMockVerdict() without the page changing.

import { mockGallery, mulberry32, FEATURE_DIM } from './mockGallery'
import { getArchive } from './archiveStore'

export const TOP_K = 3
export const SEQ_LEN = 30

// identify.py has no "unknown" cutoff yet: it always returns its closest
// match. The backend will need one. This number only fits the mock vectors;
// the real one has to be picked by testing on the real gallery.
export const MATCH_THRESHOLD = 30
// Distance that counts as "nothing alike" on the closeness bar.
export const DISTANCE_SCALE = 60

export type MockMode = 'match' | 'unknown'

export interface Suspect {
  personId: string
  distance: number
  confidence: number
}

export interface VerdictResult {
  caseId: string
  clipName: string
  framesWithPerson: number
  gallerySize: number
  suspects: Suspect[]
  isMatch: boolean
  // The clip's gait signature, kept so an unknown walker can be saved.
  queryVector: number[]
}

function euclidean(a: number[], b: number[]) {
  let sum = 0
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i]
    sum += d * d
  }
  return Math.sqrt(sum)
}

function mockQueryVector(mode: MockMode): number[] {
  if (mode === 'match') {
    // person_007's stored walk plus some noise: a second clip of the same person.
    const rand = mulberry32(7007)
    const base = mockGallery.find((e) => e.personId === 'person_007')!.featureVector
    return base.map((v) => v + (rand() - 0.5) * 0.54)
  }
  // Someone who isn't in the Archive at all.
  const rand = mulberry32(4242)
  return Array.from({ length: FEATURE_DIM }, () => (rand() - 0.5) * 2)
}

export function buildMockVerdict(mode: MockMode): VerdictResult {
  const query = mockQueryVector(mode)

  const archive = getArchive()
  const ranked = archive
    .map((entry) => ({ personId: entry.personId, distance: euclidean(query, entry.featureVector) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, TOP_K)

  const inverse = ranked.map((r) => 1 / (r.distance + 1e-6))
  const total = inverse.reduce((s, v) => s + v, 0)
  const suspects = ranked.map((r, i) => ({ ...r, confidence: (inverse[i] / total) * 100 }))

  return {
    caseId: '0042',
    clipName: mode === 'match' ? 'clip_001.mp4' : 'clip_002.mp4',
    framesWithPerson: mode === 'match' ? 87 : 64,
    gallerySize: archive.length,
    suspects,
    isMatch: suspects[0].distance <= MATCH_THRESHOLD,
    queryVector: query,
  }
}

// 0 = nothing alike, 1 = identical. Used for the closeness bar.
export function closeness(distance: number) {
  return Math.max(0, Math.min(1, 1 - distance / DISTANCE_SCALE))
}
