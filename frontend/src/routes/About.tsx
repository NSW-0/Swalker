import duckNerd from '../assets/duck_nerd.png'

export default function About() {
  return (
    <section className="page page-about-content">
      <div className="about-header">
        <h1 className="about-heading">
          About <span className="hero-brand-accent">S</span>walker
        </h1>
      </div>

      <div className="about-bubbles-scroll">
      <div className="about-bubbles">
        <article className="about-bubble">
          <span className="about-bubble-index">01</span>
          <h2 className="about-bubble-title">
            How <span className="hero-brand-accent">S</span>walker Works
          </h2>
          <ol className="about-bubble-steps">
            <li>
              <strong>Crime Scene</strong> — upload or record a walking clip of your subject.
            </li>
            <li>
              <strong>Silhouette extraction</strong> — YOLOv8 isolates the person from the
              background, frame by frame.
            </li>
            <li>
              <strong>Gait fingerprint</strong> — our model turns the walk into a
              4096-dimensional signature.
            </li>
            <li>
              <strong>Match &amp; identify</strong> — the signature is compared against every
              profile in the Archive for the closest match.
            </li>
          </ol>
        </article>

        <article className="about-bubble">
          <span className="about-bubble-index">02</span>
          <h2 className="about-bubble-title">What Is Gait Recognition?</h2>
          <p className="about-bubble-text">
            Gait recognition identifies people by <em>how</em> they walk, not by their face.
            Stride length, posture, and the rhythm of body movement form a pattern that's hard
            to disguise and easy to read from a distance, in poor lighting, or even from
            behind — no cooperation and no clear face shot required. That makes it a useful
            complement to face recognition in surveillance and forensic work.
          </p>
        </article>

        <article className="about-bubble">
          <span className="about-bubble-index">03</span>
          <h2 className="about-bubble-title">Our Model</h2>
          <p className="about-bubble-text">
            Swalker runs on a custom model, <strong>Combined GLFE + MSTFE</strong>, that fuses
            rich spatial detail of body parts with a multi-scale view of slow and fast walking
            dynamics over time. Trained on the CASIA-B benchmark at roughly 6.5M parameters, it
            reaches <strong>99.18%</strong> accuracy on normal walking — state-of-the-art among
            silhouette-based methods — and <strong>92.44%</strong> when the subject is carrying
            a bag. Recognizing someone after a clothing change is still the hardest case
            (<strong>72.36%</strong>), largely because so few examples of that exist to learn
            from.
          </p>
          <p className="about-bubble-credit">Research &amp; model by Nabeel Sawafta, 2025–2026.</p>
        </article>
      </div>
      </div>

      <img src={duckNerd} className="about-duck" alt="" />
    </section>
  )
}
