// Typed calls to the Swalker backend (backend/app/api). In development Vite proxies /api to
// http://localhost:8000, see vite.config.ts.

export const FEATURE_DIM = 4096

export interface Suspect {
  personId: string
  distance: number
  // Relative between the suspects shown (they add up to 100). Not a probability.
  confidence: number
  thumbnailUrl: string | null
  silhouetteUrls: string[]
}

export interface VerdictResult {
  caseId: string
  queryId: string
  clipName: string
  framesTotal: number
  framesWithPerson: number
  framesUsed: number
  gallerySize: number
  threshold: number
  thresholdCalibrated: boolean
  isMatch: boolean
  suspects: Suspect[] // closest first, at most 3
  querySilhouettes: string[] // PNG data URLs
}

// Identify runs as a job on the server; the Verdict page polls its status (HANDOFF §7.3).
export type JobStage = 'queued' | 'detecting' | 'silhouettes' | 'embedding' | 'searching' | 'done' | 'error'

export interface JobStatus {
  jobId: string
  status: JobStage
  progress: number // 0..1 over the whole investigation
  result?: VerdictResult
  error?: { error: string; message: string }
}

export interface CaseFile {
  personId: string
  videoPath: string
  addedAt: string
  vectorPreview: number[]
  l2Norm: number
  featureDim: number
  thumbnailUrl: string | null
}

export interface CaseFileDetail extends CaseFile {
  silhouetteUrls: string[]
}

export interface Health {
  modelLoaded: boolean
  loadError: string | null
  device: string
  gallerySize: number
  galleryCompatible: boolean
  threshold: number
  thresholdCalibrated: boolean
  maxUploadMb: number
  videoExtensions: string[]
}

// What the UI says for each backend error code. Anything else falls back to the server's message.
const FRIENDLY: Record<string, string> = {
  no_person: 'No one walks through this clip. Try a video where the whole body is in view.',
  too_few_frames: 'The walker is only on screen for a moment. Use a clip with a few seconds of walking.',
  unreadable_video: "This file couldn't be opened as a video.",
  unsupported_type: "That file type isn't a video we can read.",
  file_too_large: 'That video is too large.',
  model_not_loaded: "The detective isn't in yet: the model hasn't loaded on the server.",
  gallery_incompatible: 'The Archive was built with different settings and has to be rebuilt.',
  query_expired: 'This investigation has gone cold. Investigate the clip again to save it.',
  name_taken: 'A case file with this name already exists.',
  job_not_found: 'This investigation was lost (the server may have restarted). Investigate the clip again.',
  server_error: 'Something went wrong on the server while investigating this clip.',
  network: "Can't reach the Swalker server. Is the backend running?",
}

export class ApiError extends Error {
  readonly code: string
  readonly status: number

  constructor(code: string, message: string, status: number) {
    super(FRIENDLY[code] ?? message)
    this.code = code
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, init)
  } catch {
    throw new ApiError('network', 'Network error', 0)
  }
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    throw new ApiError(body?.error ?? 'server_error', body?.message ?? `Server error (${res.status})`, res.status)
  }
  return body as T
}

function videoForm(video: File, extra: Record<string, string> = {}) {
  const form = new FormData()
  for (const [k, v] of Object.entries(extra)) form.append(k, v)
  form.append('video', video)
  return form
}

export const api = {
  health: () => request<Health>('/api/health'),
  gallery: () => request<{ entries: CaseFile[]; compatible: boolean }>('/api/gallery'),
  caseFile: (personId: string) => request<CaseFileDetail>(`/api/gallery/${encodeURIComponent(personId)}`),
  startIdentify: (video: File) =>
    request<{ jobId: string; status: JobStage }>('/api/identify', { method: 'POST', body: videoForm(video) }),
  identifyStatus: (jobId: string) => request<JobStatus>(`/api/identify/${encodeURIComponent(jobId)}/status`),
  saveQuery: (name: string, queryId: string) =>
    request<CaseFileDetail>('/api/gallery', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, queryId }),
    }),
  addFromVideo: (name: string, video: File) =>
    request<CaseFileDetail>('/api/gallery/from-video', { method: 'POST', body: videoForm(video, { name }) }),
  renameCaseFile: (personId: string, name: string) =>
    request<CaseFileDetail>(`/api/gallery/${encodeURIComponent(personId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }),
  deleteCaseFile: (personId: string) =>
    request<null>(`/api/gallery/${encodeURIComponent(personId)}`, { method: 'DELETE' }),
}

// Same limits the backend enforces; used to warn before uploading.
export const DEFAULT_LIMITS = { maxUploadMb: 100, videoExtensions: ['.mp4', '.mov', '.avi', '.mkv', '.webm'] }

export function checkVideo(file: File, limits = DEFAULT_LIMITS): string | null {
  const dot = file.name.lastIndexOf('.')
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : ''
  if (!limits.videoExtensions.includes(ext)) {
    return `${file.name} isn't a supported video (${limits.videoExtensions.join(', ')}).`
  }
  if (file.size > limits.maxUploadMb * 1024 * 1024) {
    return `${file.name} is larger than ${limits.maxUploadMb} MB.`
  }
  return null
}

export function formatSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

// Where a distance sits on a 0..1 scale: 0 = identical, 0.5 = the match line, 1 = twice as far.
export function scalePosition(distance: number, threshold: number) {
  return Math.max(0, Math.min(1, distance / (2 * threshold)))
}
