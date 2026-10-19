// Everything the Archive holds: the mock gallery plus any walks saved from
// the Verdict page. Saved walks live in memory only, so they disappear on a
// page reload. Once the backend exists, saving becomes an API call that adds
// the person to gallery.pkl for real.

import { mockGallery } from './mockGallery'
import type { GalleryEntry } from './mockGallery'

const savedEntries: GalleryEntry[] = []

export function getArchive(): GalleryEntry[] {
  return [...mockGallery, ...savedEntries]
}

export function saveToArchive(entry: GalleryEntry) {
  savedEntries.push(entry)
}

export function isNameTaken(personId: string) {
  return getArchive().some((e) => e.personId === personId)
}

export function nextPersonId() {
  return `person_${String(getArchive().length + 1).padStart(3, '0')}`
}
