import Link from "next/link";
import { CountUp, Reveal } from "./Reveal";

/* Numbers are the real ones from the dataset and the retrieval layer, phrased for a
   visitor rather than an engineer - no chunk counts, no model names. */
const STATS = [
  { value: 391, label: "dog breeds", note: "every breed in the dataset" },
  { value: 30, label: "measured traits", note: "shedding, barking, energy, size…" },
  { value: 100, suffix: "%", label: "grounded answers", note: "every rating comes from the data" },
];

const STEPS = [
  {
    n: "01",
    title: "Tell it about your life",
    body: "A flat with thin walls. Nine hours out of the house. Two small kids and no time for grooming. Say it however you'd say it to a friend.",
  },
  {
    n: "02",
    title: "It filters, it doesn't guess",
    body: "Your requirements become real constraints against real columns. \"Won't bark much\" is a number. Breeds either meet it or they don't.",
  },
  {
    n: "03",
    title: "You get options, and the reasons",
    body: "Two or three breeds, each with the actual figures behind it, and a straight answer when nothing quite fits.",
  },
];

const FEATURES = [
  {
    title: "It picks its own approach",
    body: "Some questions are measurements. Some are descriptions. It works out which kind you asked and searches accordingly, instead of forcing everything through one method.",
  },
  {
    title: "Exact filters, not fuzzy matches",
    body: "\"Under 12kg\" and \"barely barks\" become real thresholds checked against real values. A breed either clears them or it doesn't, so nothing slips through on vague resemblance.",
  },
  {
    title: "Whole profiles when it can",
    body: "Once the list is short, it reads each breed's complete write up rather than a few fragments. Nothing relevant gets left behind because a snippet was cut in the wrong place.",
  },
  {
    title: "It reads the fine print too",
    body: "Ask about hip dysplasia, allergies or living with cats and it searches the full text of all 391 breeds, because those answers live in the writing rather than in any column.",
  },
  {
    title: "It says when nothing fits",
    body: "If no breed meets every requirement, it says so, loosens the least important one, and tells you exactly which. You get honest near misses instead of a confident wrong answer.",
  },
  {
    title: "Every figure is traceable",
    body: "Each rating it quotes came from the dataset, not from the model's memory. You can open the trace on any answer and see precisely what was looked up.",
  },
];

export default function Landing() {
  return (
    <main className="landing">
      <section className="hero">
        <div className="hero-inner">
          <Reveal as="div" className="eyebrow">
            🐾 PawGPT
          </Reveal>
          <Reveal as="h1" delay={80}>
            Find the dog that fits
            <br />
            <em>your actual life.</em>
          </Reveal>
          <Reveal as="p" delay={160} className="lede">
            Not the one that looks best in photos. Describe your home, your hours and
            your patience, and get breeds that genuinely match, with the numbers to
            back it up.
          </Reveal>
          <Reveal delay={240}>
            <Link href="/chat" className="cta">
              Let&apos;s find your breed
              <span aria-hidden>→</span>
            </Link>
          </Reveal>
          <Reveal delay={320} className="hero-hint">
            Free · no sign-up · answers in seconds
          </Reveal>
        </div>
        <div className="hero-glow" aria-hidden />
      </section>

      <section className="stats">
        {STATS.map((s, i) => (
          <Reveal key={s.label} className="stat" delay={i * 90}>
            <div className="stat-value">
              <CountUp value={s.value} suffix={s.suffix ?? ""} />
            </div>
            <div className="stat-label">{s.label}</div>
            <div className="stat-note">{s.note}</div>
          </Reveal>
        ))}
      </section>

      <section className="section">
        <Reveal as="h2">How it works</Reveal>
        <div className="steps">
          {STEPS.map((step, i) => (
            <Reveal key={step.n} className="step" delay={i * 110}>
              <div className="step-n">{step.n}</div>
              <h3>{step.title}</h3>
              <p>{step.body}</p>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="section">
        <Reveal as="h2">Why it&apos;s different</Reveal>
        <Reveal as="p" className="section-lede" delay={70}>
          It doesn&apos;t answer from memory. It decides what to look up, looks it up,
          and answers only from what it found.
        </Reveal>
        <div className="features">
          {FEATURES.map((f, i) => (
            <Reveal key={f.title} className="feature" delay={(i % 2) * 90}>
              <h3>{f.title}</h3>
              <p>{f.body}</p>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="section closing">
        <Reveal as="h2">Ready when you are.</Reveal>
        <Reveal as="p" className="section-lede" delay={70}>
          One question is usually enough to get somewhere useful.
        </Reveal>
        <Reveal delay={140}>
          <Link href="/chat" className="cta">
            Let&apos;s find your breed
            <span aria-hidden>→</span>
          </Link>
        </Reveal>
      </section>

      <footer className="foot">
        Recommendations come from a public breed dataset and are a starting point, not
        veterinary advice.
      </footer>
    </main>
  );
}
