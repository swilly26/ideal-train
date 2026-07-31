import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "ApexTrade — Peak trading performance. Powered by AI." },
      { name: "description", content: "Adaptive automated trading strategies built for disciplined execution." },
    ],
  }),
  component: Home,
});

const steps = [
  {
    number: "01",
    title: "Connect your brokerage",
    description: "Link your Alpaca account in minutes. Your credentials stay protected, and you keep full control of your capital.",
    icon: "↗",
  },
  {
    number: "02",
    title: "AI strategies run automatically",
    description: "ApexTrade adapts to market conditions with mean reversion, momentum, and liquidity sweep strategies.",
    icon: "✦",
  },
  {
    number: "03",
    title: "You collect profits",
    description: "Every trade is stop-loss protected, with positions automatically liquidated at the end of the day.",
    icon: "◒",
  },
];

const plans = [
  {
    name: "Starter",
    price: "$49",
    description: "A focused start for hands-off traders.",
    features: ["2 strategies", "5 symbols", "Daily summaries"],
    stripeLink: "https://buy.stripe.com/cNi5kC4f93zwerS5aV5wI00",
  },
  {
    name: "Pro",
    price: "$99",
    description: "The complete adaptive trading toolkit.",
    features: ["All strategies", "15 symbols", "Real-time dashboard", "Priority execution"],
    featured: true,
    stripeLink: "https://buy.stripe.com/aFa8wO12Xgmi6ZqcDn5wI01",
  },
  {
    name: "Turbo",
    price: "$199",
    description: "More range, more control, more opportunity.",
    features: ["Everything in Pro", "Leveraged ETF strategies", "50% allocation mode", "Dedicated support"],
    stripeLink: "https://buy.stripe.com/aFa00ih1V6LIabCbzj5wI02",
  },
];

function Home() {
  return (
    <main className="apextrade-page">
      <nav className="site-nav container">
        <a className="brand" href="#top" aria-label="ApexTrade home"><span className="brand-mark">◈</span>Apex<span>Trade</span></a>
        <div className="nav-links"><a href="#how-it-works">How it works</a><a href="#performance">Performance</a><a href="#pricing">Pricing</a></div>
        <div className="nav-links"><a href="/login">Log in</a><a href="/signup">Sign up</a></div>
        <a className="nav-cta" href="#pricing">Get started <span>↗</span></a>
      </nav>

      <section className="hero container" id="top">
        <div className="hero-copy">
          <div className="eyebrow"><span className="pulse-dot" /> Intelligent trading, simplified</div>
          <h1>AI-powered day trading that <em>works while you sleep.</em></h1>
          <p className="hero-sub">Automated strategies that continuously adapt to market conditions—so you can pursue opportunities without watching the screen.</p>
          <div className="hero-actions"><a className="button button-primary" href="https://buy.stripe.com/aFa8wO12Xgmi6ZqcDn5wI01" target="_blank" rel="noopener noreferrer">Start trading smarter <span>↗</span></a><a className="text-link" href="#how-it-works">See how it works <span>↓</span></a></div>
          <p className="hero-note"><span>✓</span> Built for disciplined, hands-off execution</p>
        </div>
        <div className="hero-visual" aria-label="Live strategy performance preview">
          <div className="glow" />
          <div className="terminal-card">
            <div className="terminal-top"><span className="live"><i /> LIVE STRATEGY ENGINE</span><span className="dots">•••</span></div>
            <div className="chart-label">Portfolio value <strong>$124,842.16</strong><small>+18.42% <span>↑</span></small></div>
            <div className="chart"><div className="chart-grid" /><svg viewBox="0 0 440 180" preserveAspectRatio="none" role="img" aria-label="Upward performance chart"><defs><linearGradient id="chartFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#63e6be" stopOpacity=".3"/><stop offset="1" stopColor="#63e6be" stopOpacity="0"/></linearGradient></defs><path d="M0 155 C25 150 30 135 52 142 S76 120 94 130 S122 107 143 118 S170 78 190 98 S212 80 235 88 S255 53 277 68 S302 46 321 55 S344 30 365 45 S400 20 440 10 V180 H0Z" fill="url(#chartFill)"/><path d="M0 155 C25 150 30 135 52 142 S76 120 94 130 S122 107 143 118 S170 78 190 98 S212 80 235 88 S255 53 277 68 S302 46 321 55 S344 30 365 45 S400 20 440 10" fill="none" stroke="#63e6be" strokeWidth="3"/></svg></div>
            <div className="chart-times"><span>9:30 AM</span><span>11:00 AM</span><span>1:00 PM</span><span>4:00 PM</span></div>
            <div className="trade-row"><span className="trade-icon">↗</span><span><b>Momentum / NVDA</b><small>Position closed · 2m ago</small></span><strong>+$428.60</strong></div>
          </div>
          <div className="float-card"><span className="float-icon">✦</span><span><small>AI confidence</small><b>94.8%</b></span><span className="up">↑</span></div>
        </div>
      </section>

      <section className="proof-strip"><div className="container proof-inner"><span>POWERED BY ADAPTIVE INTELLIGENCE</span><span>ALPACA INTEGRATION</span><span>RISK-FIRST EXECUTION</span></div></section>

      <section className="section container" id="how-it-works"><div className="section-heading"><div><div className="eyebrow">The simple path to smarter trading</div><h2>From setup to strategy.<br /><em>Fully automated.</em></h2></div><p>ApexTrade brings institutional-style discipline to your trading day, without the complexity.</p></div><div className="steps">{steps.map((step) => <article className="step" key={step.number}><div className="step-top"><span className="step-number">{step.number}</span><span className="step-icon">{step.icon}</span></div><h3>{step.title}</h3><p>{step.description}</p></article>)}</div></section>

      <section className="performance" id="performance"><div className="container"><div className="section-heading performance-heading"><div><div className="eyebrow">Designed for consistency</div><h2>Performance you can<br /><em>measure.</em></h2></div><p>Clear signals, clear reporting, and risk controls at every level.</p></div><div className="stats"><div><strong>78<span>%</span></strong><label>Win rate</label></div><div><strong>~1.5<span>%</span></strong><label>Avg. return per trade</label></div><div><strong>24<span>/7</span></strong><label>Risk monitoring</label></div><div><strong>0<span> overnight</span></strong><label>Open positions</label></div></div><p className="disclaimer">Paper trading results — real performance may vary</p></div></section>

      <section className="section pricing-section container" id="pricing"><div className="pricing-heading"><div className="eyebrow">Choose your edge</div><h2>Plans that scale with<br /><em>your ambition.</em></h2><p>Start with the essentials. Upgrade when you're ready to put more strategies to work.</p></div><div className="plans">{plans.map((plan) => <article className={`plan ${plan.featured ? "featured" : ""}`} key={plan.name}>{plan.featured && <div className="popular">MOST POPULAR</div>}<h3>{plan.name}</h3><p>{plan.description}</p><div className="price"><strong>{plan.price}</strong><span>/ month</span></div><a className={`button ${plan.featured ? "button-primary" : "button-outline"}`} href={plan.stripeLink} target="_blank" rel="noopener noreferrer">Get started <span>↗</span></a><ul>{plan.features.map((feature) => <li key={feature}><span>✓</span>{feature}</li>)}</ul></article>)}</div></section>

      <footer className="footer"><div className="container footer-inner"><a className="brand" href="#top"><span className="brand-mark">◈</span>Apex<span>Trade</span></a><p>© 2026 ApexTrade. Not financial advice. Trading involves risk.</p><a href="#top" className="back-top">Back to top ↑</a></div></footer>
    </main>
  );
}
