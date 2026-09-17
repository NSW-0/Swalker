// Mock stand-in for gallery.pkl (built by gallery_builder.py) until the
// real backend is wired up. Shape mirrors that pipeline on purpose:
//   gallery.pkl -> { person_ids: [...], features: (N, 4096), video_paths: [...] }
// This file just reshapes that into one record per person, JSON-friendly,
// so swapping it for a real API response later (e.g. GET /api/gallery)
// is a drop-in — nothing that reads `mockGallery` needs to change shape.

export const FEATURE_DIM = 4096

export interface GalleryEntry {
  personId: string
  videoPath: string
  featureVector: number[]
}

// Small seedable PRNG (mulberry32) so each person's mock vector is stable
// across reloads instead of reshuffling on every render.
function mulberry32(seed: number) {
  return function () {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function mockFeatureVector(seed: number, dim = FEATURE_DIM): number[] {
  const rand = mulberry32(seed)
  return Array.from({ length: dim }, () => (rand() - 0.5) * 2)
}

const PEOPLE = [
  { id: 'person_001', seed: 1001, clip: 'walk1.mp4' },
  { id: 'person_002', seed: 1002, clip: 'walk1.mp4' },
  { id: 'person_003', seed: 1003, clip: 'walk1.mp4' },
  { id: 'person_004', seed: 1004, clip: 'walk1.mp4' },
  { id: 'person_005', seed: 1005, clip: 'walk1.mp4' },
  { id: 'person_006', seed: 1006, clip: 'walk1.mp4' },
  { id: 'person_007', seed: 1007, clip: 'walk1.mp4' },
  { id: 'person_008', seed: 1008, clip: 'walk1.mp4' },
  { id: 'person_009', seed: 1009, clip: 'walk1.mp4' },
  { id: 'person_010', seed: 1010, clip: 'walk1.mp4' },
  { id: 'person_011', seed: 1011, clip: 'walk1.mp4' },
  { id: 'person_012', seed: 1012, clip: 'walk1.mp4' },
  { id: 'person_013', seed: 1013, clip: 'walk1.mp4' },
  { id: 'person_014', seed: 1014, clip: 'walk1.mp4' },
  { id: 'person_015', seed: 1015, clip: 'walk1.mp4' },
  { id: 'person_016', seed: 1016, clip: 'walk1.mp4' },
  { id: 'person_017', seed: 1017, clip: 'walk1.mp4' },
  { id: 'person_018', seed: 1018, clip: 'walk1.mp4' },
  { id: 'person_019', seed: 1019, clip: 'walk1.mp4' },
  { id: 'person_020', seed: 1020, clip: 'walk1.mp4' },
]

export const mockGallery: GalleryEntry[] = PEOPLE.map(({ id, seed, clip }) => ({
  personId: id,
  videoPath: `gallery_videos/${id}/${clip}`,
  featureVector: mockFeatureVector(seed),
}))
