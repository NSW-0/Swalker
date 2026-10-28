// Real 64x44 silhouettes from the backend: the frames GLTNet actually reads.

export function SilhouetteThumb({ src }: { src: string | null }) {
  if (!src) return <span className="sil-empty" aria-hidden="true" />
  return <img className="sil-img" src={src} alt="" />
}

export function SilhouetteStrip({ frames, className = 'verdict-strip' }: { frames: string[]; className?: string }) {
  if (frames.length === 0) {
    return <div className={`${className} sil-strip-empty`}>No silhouettes on file</div>
  }
  return (
    <div className={className}>
      {frames.map((src, i) => (
        <img key={i} className="sil-img" src={src} alt={i === 0 ? 'Walking silhouette frames' : ''} />
      ))}
    </div>
  )
}
