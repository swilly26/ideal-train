import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "ApexTrade — Automated day trading, engine paused" },
      {
        name: "description",
        content:
          "ApexTrade is building an automated day-trading engine. It is switched off while we look for a strategy with a verified edge, and no performance figures are published here.",
      },
    ],
  }),
  component: Home,
});

const steps = [
  {
    number: "01",
    title: "Connect your brokerage",
    description: "You would link your Alpaca account directly. Your capital stays in your own brokerage account — ApexTrade never holds it.",
    icon: "↗",
  },
  {
    number: "02",
    title: "Strategies run automatically",
    description: "The engine is built to place and manage trades without you watching the screen. It is switched off while we look for a strategy that survives honest testing.",
    icon: "✦",
  },
  {
    number: "03",
    title: "Results only when they are real",
    description: "We publish performance only when it comes from real broker fills and holds up on data the strategy was not fitted to. Until then, there is nothing to show.",
    icon: "◒",
  },
];

// Prices are the owner's (indicative, not on sale). No checkout link, no feature
// list: nothing here may advertise a capability or a result we do not have.
const plans = [
  {
    name: "Starter",
    price: "$49",
    description: "A focused start for hands-off traders.",
  },
  {
    name: "Pro",
    price: "$99",
    description: "The complete trading toolkit.",
    featured: true,
  },
  {
    name: "Turbo",
    price: "$199",
    description: "More range, more control, more opportunity.",
  },
];

function Home() {
  return (
    <main className="apextrade-page">
      <nav className="site-nav container">
        <a className="brand" href="#top" aria-label="ApexTrade home"><span className="brand-mark">◈</span>Apex<span>Trade</span></a>
        <div className="nav-links"><a href="#how-it-works">How it works</a><a href="#status">Status</a><a href="#pricing">Pricing</a></div>
        <div className="nav-links"><a href="/login">Log in</a><a href="/signup">Sign up</a></div>
        <a className="nav-cta" href="#pricing">See pricing <span>↓</span></a>
      </nav>

      <section className="hero container" id="top">
        <div className="hero-copy">
          <div className="eyebrow">Automated day trading, in development</div>
          <h1>Automated day trading, <em>paused until the edge is proven.</em></h1>
          <p className="hero-sub">Our strategy engine has been switched off since 23 September 2026 while we look for a trading method with a real, verified edge. We publish no performance figures here, because we do not have any we can stand behind.</p>
          <div className="hero-actions"><button type="button" className="button button-primary" disabled aria-disabled="true">No plans on sale yet</button><a className="text-link" href="#how-it-works">See how it works <span>↓</span></a></div>
          <p className="hero-note"><span>•</span> Engine paused — nothing is trading and nothing is for sale</p>
        </div>
        <div className="hero-visual" aria-label="Engine status">
          <div className="glow" />
          <div className="terminal-card">
            <div className="terminal-top"><span className="paused"><i /> ENGINE PAUSED</span><span className="dots">•••</span></div>
            <div className="status-list">
              <div className="status-row"><b>Status</b><small>Switched off deliberately on 23 Sep 2026</small></div>
              <div className="status-row"><b>Why</b><small>No strategy has shown a verified edge yet</small></div>
              <div className="status-row"><b>Results published</b><small>None — we do not have any we can stand behind</small></div>
              <div className="status-row"><b>Plans</b><small>Not on sale</small></div>
            </div>
          </div>
        </div>
      </section>

      <section className="proof-strip"><div className="container proof-inner"><span>ENGINE PAUSED SINCE 23 SEP 2026</span><span>NO PERFORMANCE FIGURES PUBLISHED</span><span>NO PLANS ON SALE YET</span></div></section>

      <section className="section container" id="how-it-works"><div className="section-heading"><div><div className="eyebrow">What the product is meant to do</div><h2>From setup to strategy.<br /><em>Not running today.</em></h2></div><p>The shape of the product, and the honest state of it.</p></div><div className="steps">{steps.map((step) => <article className="step" key={step.number}><div className="step-top"><span className="step-number">{step.number}</span><span className="step-icon">{step.icon}</span></div><h3>{step.title}</h3><p>{step.description}</p></article>)}</div></section>

      <section className="performance" id="status"><div className="container"><div className="section-heading performance-heading"><div><div className="eyebrow">Straight answer</div><h2>No performance figures<br /><em>are published here.</em></h2></div><p>Not because we are being coy — because we do not have any we can stand behind.</p></div><div className="stats"><div><strong>Paused</strong><label>Strategy engine, off since 23 Sep 2026</label></div><div><strong>None</strong><label>Performance results published on this site</label></div><div><strong>Unproven</strong><label>No strategy has a verified edge yet</label></div><div><strong>Not yet</strong><label>Plans for sale</label></div></div><p className="disclaimer">We will publish results only when they are measured from broker fills and validated on data the strategy was not fitted to. Nothing on this page is a promise, a projection or a guarantee of profit.</p></div></section>

      <section className="section pricing-section container" id="pricing"><div className="pricing-heading"><div className="eyebrow">Planned pricing — not on sale</div><h2>Plans we have priced,<br /><em>but cannot yet sell.</em></h2><p>The prices below are indicative and set by the owner. Nothing on this site can be bought today.</p></div><div className="plans">{plans.map((plan) => <article className={`plan ${plan.featured ? "featured" : ""}`} key={plan.name}><h3>{plan.name}</h3><p>{plan.description}</p><div className="price"><strong>{plan.price}</strong><span>indicative</span></div><button type="button" className={`button ${plan.featured ? "button-primary" : "button-outline"}`} disabled aria-disabled="true">Not on sale yet</button></article>)}</div><p className="disclaimer pricing-note">No plan is available for purchase. Our payment account cannot take recurring subscriptions yet, the trading engine is paused, and we will not sell access before a strategy has a verified edge.</p></section>

      <footer className="footer"><div className="container footer-inner"><a className="brand" href="#top"><span className="brand-mark">◈</span>Apex<span>Trade</span></a><p>© 2026 ApexTrade. Not financial advice. Trading involves risk.</p><a href="#top" className="back-top">Back to top ↑</a></div></footer>
    </main>
  );
}
