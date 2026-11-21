(() => {
  "use strict";

  const BUCKETS = [
    { key: "alive", label: "Alive", color: "var(--alive)" },
    { key: "drifted", label: "Drifted", color: "var(--drifted)" },
    { key: "dead", label: "Dead", color: "var(--dead)" },
  ];
  const FATES = {
    intact: ["alive", "Intact", "Same address, essentially the same text."],
    edited: ["alive", "Edited", "Same address, text revised but recognisably the same page."],
    moved: ["alive", "Moved", "Redirects to a new address that serves the same page."],
    reachable: ["alive", "Reachable", "Answers normally, but the content could not be compared (a PDF, an image, very short text)."],
    replaced: ["drifted", "Replaced", "Same address now serves different text."],
    redirect_home: ["dead", "Sent home", "A deep link that now redirects to a site's front page."],
    soft_404: ["dead", "Soft 404", "Answers OK but says the page is gone."],
    not_found: ["dead", "Not found", "HTTP 404, 410 or another client error."],
    parked: ["dead", "Parked", "The domain is parked or for sale."],
    defunct: ["dead", "Defunct", "A server default page, suspended account, or expired site."],
    domain_gone: ["dead", "Domain gone", "The domain name no longer resolves at all."],
    blocked: ["unknown", "Blocked", "Refused our visit (403, 429, a bot challenge). Left out of shares."],
    server_error: ["unknown", "Server error", "A 5xx error. Left out of shares."],
    unreachable: ["unknown", "Unreachable", "Timed out, refused the connection, or failed TLS. Left out of shares."],
  };
  const BUCKET_COLOR = { alive: "var(--alive)", drifted: "var(--drifted)", dead: "var(--dead)", unknown: "var(--unknown)" };

  const $ = (id) => document.getElementById(id);
  const show = (id) => { $(id).hidden = false; };
  const pct = (x, digits = 0) => (x == null ? "–" : `${(x * 100).toFixed(digits)}%`);
  const num = (x) => (x == null ? "–" : x.toLocaleString("en-US"));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtDate = (iso) => (iso ? new Date(iso + "T00:00:00Z").toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" }) : "");

  function roman(n) {
    const map = [[1000, "M"], [900, "CM"], [500, "D"], [400, "CD"], [100, "C"], [90, "XC"], [50, "L"], [40, "XL"], [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"]];
    let out = "";
    for (const [v, s] of map) while (n >= v) { out += s; n -= v; }
    return out;
  }

  // ── tooltip ────────────────────────────────────────────
  const tip = $("tooltip");
  function showTip(html, x, y) {
    tip.innerHTML = html;
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let left = x + 14, top = y + 14;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
    tip.style.left = `${Math.max(8, left)}px`;
    tip.style.top = `${Math.max(8, top)}px`;
  }
  const hideTip = () => { tip.hidden = true; };
  const tipRow = (color, label, value) =>
    `<div class="tt-row"><span class="swatch" style="background:${color}"></span>${esc(label)} <strong>${esc(value)}</strong></div>`;

  function legend(el, items) {
    el.innerHTML = items.map(([cls, color, label]) =>
      `<span><span class="swatch ${cls}" style="background:${color}"></span>${esc(label)}</span>`).join("");
  }

  async function getJSON(path) {
    const res = await fetch(path, { cache: "no-cache" });
    if (!res.ok) throw new Error(`${path}: ${res.status}`);
    return res.json();
  }

  // ── hero ───────────────────────────────────────────────
  function renderHero(s) {
    const h = s.headline;
    $("h-dead").textContent = pct(h.dead_share);
    const oldest = s.by_year.find((y) => y.known >= 100);
    const oldestLine = oldest && oldest.dead_share != null
      ? ` For pages first captured in ${oldest.year}, it is <strong>${pct(oldest.dead_share)}</strong>.` : "";
    $("h-dead-caption").innerHTML =
      `of <strong>${num(s.totals.known)}</strong> web pages with a known fate no longer exist.${oldestLine}`;
    const half = s.decay.survival, unch = s.decay.unchanged;
    $("h-half").textContent = half ? half.half_life_years.toFixed(1) : "–";
    $("h-unchanged").textContent = unch ? unch.half_life_years.toFixed(1) : "–";
    $("h-domain").textContent = pct(h.domain_gone_share);
    $("h-lost").textContent = pct(s.archive.lost_share_of_known);
    const first = s.sample.first_crawl_date ? s.sample.first_crawl_date.slice(0, 4) : "";
    const last = s.sample.last_crawl_date ? s.sample.last_crawl_date.slice(0, 4) : "";
    $("lede").innerHTML =
      `${num(s.sample.urls)} web pages captured by Common Crawl between ${first} and ${last}, from ${num(s.sample.hosts)} sites, ` +
      `revisited on <strong>${fmtDate(s.census_date)}</strong> to see which still exist, which changed, and which are gone.`;
    show("hero");
  }

  // ── years: 100% stacked bars ───────────────────────────
  function renderYears(s) {
    legend($("legend-years"), [
      ...BUCKETS.map((b) => ["", b.color, b.label]),
    ]);
    const rows = s.by_year.filter((y) => y.known > 0);
    const chart = $("chart-years");
    chart.innerHTML = rows.map((y) => {
      const segs = BUCKETS.map((b) => {
        const share = y.buckets[b.key] / y.known;
        if (!share) return "";
        const label = share >= 0.09 ? `<span class="seg-label">${pct(share)}</span>` : "";
        return `<div class="seg" data-year="${y.year}" data-bucket="${b.key}" style="flex:${share} 1 0;background:${b.color}">${label}</div>`;
      }).join("");
      return `<div class="stack-row"><div class="stack-label">${y.year}<small>${roman(y.year)}</small></div>
        <div class="stack-bar">${segs}</div><div class="stack-unknown">${pct(y.unknown_share)}</div></div>`;
    }).join("") +
      `<div class="stack-axis"><span></span><div class="ticks"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div><span class="unk-head">unknown</span></div>`;

    const byYear = Object.fromEntries(rows.map((y) => [y.year, y]));
    chart.onmousemove = (e) => {
      const seg = e.target.closest(".seg");
      if (!seg) return hideTip();
      const y = byYear[seg.dataset.year], b = seg.dataset.bucket;
      const parts = Object.entries(y.statuses)
        .filter(([k]) => FATES[k] && FATES[k][0] === b)
        .sort((a, c) => c[1] - a[1])
        .map(([k, n]) => `<div>${esc(FATES[k][1])}: ${num(n)}</div>`).join("");
      showTip(`<div class="tt-title">Captured ${y.year} · ${num(y.known)} judged</div>` +
        tipRow(BUCKET_COLOR[b], BUCKETS.find((x) => x.key === b).label, `${pct(y.buckets[b] / y.known, 1)} (${num(y.buckets[b])})`) +
        parts, e.clientX, e.clientY);
    };
    chart.onmouseleave = hideTip;

    $("table-years").innerHTML = `<table class="data-table"><thead><tr><th>Year</th><th>Judged</th><th>Alive</th><th>Drifted</th><th>Dead</th><th>Dead 95% CI</th><th>Unknown</th></tr></thead><tbody>` +
      rows.map((y) => `<tr><td>${y.year}</td><td>${num(y.known)}</td><td>${pct(y.alive_share, 1)}</td><td>${pct(y.buckets.drifted / y.known, 1)}</td><td>${pct(y.dead_share, 1)}</td><td>${pct(y.dead_ci[0], 1)} to ${pct(y.dead_ci[1], 1)}</td><td>${pct(y.unknown_share, 1)}</td></tr>`).join("") +
      `</tbody></table>`;
    show("years");
  }

  // ── survival curve ─────────────────────────────────────
  function renderSurvival(s) {
    const pts = s.by_year.filter((y) => y.known >= 30 && y.survival != null);
    if (!pts.length) return;
    const el = $("chart-survival");
    show("survival");
    const W = Math.max(320, Math.round(el.clientWidth || 800));
    const H = W < 600 ? 280 : 340, m = { t: 16, r: W < 600 ? 12 : 24, b: 40, l: 44 };
    const maxAge = Math.max(14, Math.ceil(Math.max(...pts.map((p) => p.age_years)) + 1));
    const x = (t) => m.l + (t / maxAge) * (W - m.l - m.r);
    const y = (v) => m.t + (1 - v) * (H - m.t - m.b);
    const fitPath = (lam) => {
      let d = "";
      for (let t = 0; t <= maxAge; t += maxAge / 120) d += `${d ? "L" : "M"}${x(t).toFixed(1)},${y(Math.exp(-lam * t)).toFixed(1)}`;
      return d;
    };
    const surv = s.decay.survival, unch = s.decay.unchanged;
    let grid = "";
    for (const v of [0, 0.25, 0.5, 0.75, 1]) grid += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`;
    let axis = "";
    for (let t = 0; t <= maxAge; t += W < 600 ? 4 : 2) axis += `<text x="${x(t)}" y="${H - m.b + 18}" text-anchor="middle">${t}</text>`;
    axis += `<text x="${(m.l + W - m.r) / 2}" y="${H - 4}" text-anchor="middle">age of page at revisit, years</text>`;

    let guides = "";
    if (surv && surv.half_life_years <= maxAge) {
      const hx = x(surv.half_life_years);
      const anchor = hx > W * 0.6 ? "end" : "start";
      guides += `<line class="half-guide" x1="${m.l}" x2="${hx}" y1="${y(0.5)}" y2="${y(0.5)}"/><line class="half-guide" x1="${hx}" x2="${hx}" y1="${y(0.5)}" y2="${y(0)}"/>` +
        `<text class="half-label" x="${anchor === "end" ? hx - 8 : hx + 8}" y="${y(0.5) + 18}" text-anchor="${anchor}">half gone at ${surv.half_life_years.toFixed(1)} yrs</text>`;
    }
    const band = pts.map((p) => {
      const lo = 1 - p.dead_ci[1], hi = 1 - p.dead_ci[0];
      return `<rect class="ci-band" x="${x(p.age_years) - 5}" width="10" y="${y(hi)}" height="${Math.max(1, y(lo) - y(hi))}" rx="2"/>`;
    }).join("");
    const dots = pts.map((p) => `<circle class="obs" cx="${x(p.age_years)}" cy="${y(p.survival)}" r="5"/>`).join("");
    const dots2 = pts.filter((p) => p.unchanged != null).map((p) => `<circle class="obs-unchanged" cx="${x(p.age_years)}" cy="${y(p.unchanged)}" r="4"/>`).join("");

    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Share of pages still present by age, with exponential fits">
      <g class="grid">${grid}</g><g class="axis">${axis}</g>
      ${band}${guides}
      ${surv ? `<path class="fit" d="${fitPath(surv.lambda_per_year)}"/>` : ""}
      ${unch ? `<path class="fit-unchanged" d="${fitPath(unch.lambda_per_year)}"/>` : ""}
      ${dots}${dots2}
      <line class="crosshair" id="xhair" y1="${m.t}" y2="${H - m.b}" visibility="hidden"/>
      <rect id="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>
    </svg>`;
    legend($("legend-survival"), [
      ["dot", "var(--alive)", "Still present (alive or drifted), by capture year"],
      ["line", "var(--dead)", "Exponential fit"],
      ["dot", "var(--drifted)", "Still the same page (intact, edited, moved)"],
      ["line", "var(--drifted)", "Fit, same page"],
    ]);
    if (surv) $("survival-sub").textContent =
      `Share of pages still present, by how old they were when we revisited. On this fit, half of all pages are gone after ${surv.half_life_years.toFixed(1)} years` +
      (unch ? `, and half no longer carry their original wording after ${unch.half_life_years.toFixed(1)} years.` : ".");

    const svg = el.querySelector("svg"), xhair = svg.querySelector("#xhair");
    svg.querySelector("#hit").addEventListener("mousemove", (e) => {
      const r = svg.getBoundingClientRect();
      const age = ((e.clientX - r.left) * (W / r.width) - m.l) / (W - m.l - m.r) * maxAge;
      const p = pts.reduce((a, b) => (Math.abs(b.age_years - age) < Math.abs(a.age_years - age) ? b : a));
      xhair.setAttribute("x1", x(p.age_years)); xhair.setAttribute("x2", x(p.age_years));
      xhair.setAttribute("visibility", "visible");
      showTip(`<div class="tt-title">Captured ${p.year} · ${p.age_years.toFixed(1)} yrs old</div>` +
        tipRow("var(--alive)", "Still present", pct(p.survival, 1)) +
        tipRow("var(--drifted)", "Same page", pct(p.unchanged, 1)) +
        tipRow("var(--dead)", "Dead", `${pct(p.dead_share, 1)} (${pct(p.dead_ci[0], 0)} to ${pct(p.dead_ci[1], 0)})`) +
        `<div>${num(p.known)} pages judged</div>`, e.clientX, e.clientY);
    });
    svg.querySelector("#hit").addEventListener("mouseleave", () => { hideTip(); xhair.setAttribute("visibility", "hidden"); });
    show("survival");
  }

  // ── horizontal bars ────────────────────────────────────
  function hbars(el, items, max) {
    el.innerHTML = items.map((it) => {
      const w = (it.value / max) * 100;
      const ci = it.ci ? `<span class="hbar-ci" style="left:${(it.ci[0] / max) * 100}%;width:${((it.ci[1] - it.ci[0]) / max) * 100}%"></span>` : "";
      return `<div class="hbar" data-tip="${esc(it.tip || "")}"><span class="hbar-name" title="${esc(it.name)}">${esc(it.name)}</span>
        <div class="hbar-track"><span class="hbar-fill" style="width:${w}%"></span>${ci}</div>
        <span class="hbar-value">${pct(it.value)}</span></div>`;
    }).join("");
    el.onmousemove = (e) => {
      const row = e.target.closest(".hbar");
      if (!row || !row.dataset.tip) return hideTip();
      showTip(row.dataset.tip, e.clientX, e.clientY);
    };
    el.onmouseleave = hideTip;
  }

  function renderCauses(s) {
    const items = Object.entries(s.dead_breakdown)
      .map(([k, v]) => ({ name: FATES[k] ? FATES[k][1] : k, value: v, tip: esc(FATES[k] ? FATES[k][2] : k) }))
      .sort((a, b) => b.value - a.value);
    if (!items.length) return;
    hbars($("chart-causes"), items, Math.max(...items.map((i) => i.value)));
    show("causes");
  }

  function renderTlds(s) {
    if (!s.by_tld.length) return;
    const items = s.by_tld.map((t) => ({
      name: `.${t.tld}`, value: t.dead_share, ci: t.dead_ci,
      tip: `.${esc(t.tld)} · ${num(t.known)} judged<br>dead ${pct(t.dead_share, 1)} (95% CI ${pct(t.dead_ci[0], 1)} to ${pct(t.dead_ci[1], 1)})`,
    }));
    hbars($("chart-tlds"), items, Math.min(1, Math.max(...items.map((i) => i.ci[1] || i.value)) * 1.05));
    show("tlds");
  }

  function renderArchive(s) {
    const a = s.archive;
    if (!a.dead_checked) return;
    $("archive-text").innerHTML =
      `We asked the Wayback Machine about <strong>${num(a.dead_checked)}</strong> randomly chosen dead pages. ` +
      `<strong>${pct(a.dead_archived_share)}</strong> have at least one archived copy (95% CI ${pct(a.dead_archived_ci[0])} to ${pct(a.dead_archived_ci[1])}). ` +
      `The rest are gone with no copy we can find: roughly <strong>${pct(a.lost_share_of_known, 1)}</strong> of every page we could judge.`;
    show("archive");
  }

  // ── graveyard ──────────────────────────────────────────
  function renderGraveyard(graves, s) {
    if (!graves.length) return;
    const select = $("grave-year");
    const years = [...new Set(graves.map((g) => (g.crawl_date || "").slice(0, 4)).filter(Boolean))].sort();
    select.innerHTML = `<option value="">All years</option>` + years.map((y) => `<option value="${y}">${y}</option>`).join("");
    const census = fmtDate(s.census_date);
    const draw = () => {
      const yr = select.value;
      const pool = graves.filter((g) => !yr || (g.crawl_date || "").startsWith(yr));
      const pick = [];
      const copy = pool.slice();
      while (pick.length < 12 && copy.length) pick.push(copy.splice(Math.floor(Math.random() * copy.length), 1)[0]);
      $("stones").innerHTML = pick.map((g) => {
        let path = "";
        try { const u = new URL(g.url); path = (u.pathname + u.search).slice(0, 60); } catch (_) { /* keep empty */ }
        const year = (g.crawl_date || "").slice(0, 4);
        const fate = FATES[g.status] ? FATES[g.status][1] : g.status;
        return `<article class="stone">
          <div class="cross">+</div>
          <div class="host">${esc(g.host)}</div>
          <div class="path">${esc(path)}</div>
          <div class="roman">${year ? roman(+year) : ""}</div>
          <div class="dates">captured ${esc(fmtDate(g.crawl_date))}<br>found dead ${esc(census)}</div>
          <div class="words">${g.last_words ? `“${esc(g.last_words)}”` : ""}</div>
          <div class="cause">${esc(fate)}</div>
          <a href="${esc(g.wayback)}" target="_blank" rel="noopener nofollow">visit the ghost ↗</a>
        </article>`;
      }).join("");
    };
    select.onchange = draw;
    $("grave-shuffle").onclick = draw;
    draw();
    show("graveyard");
  }

  function renderMethod(s) {
    $("defs").innerHTML = Object.entries(FATES).map(([, [bucket, label, text]]) =>
      `<tr><td><span class="tag" style="background:${BUCKET_COLOR[bucket]}"></span>${esc(label)}</td><td>${esc(text)}</td></tr>`).join("");
    if (s) {
      $("m-sample").textContent =
        `${num(s.sample.urls)} pages from ${s.sample.crawls} Common Crawl snapshots collected by this project, up to 1,000 per snapshot, ` +
        `chosen by a salted hash of the URL so the same pages are chosen every time. A URL that appears in several crawls counts once, in its earliest crawl. ` +
        `The crawl collector kept English pages only. Content thresholds: intact up to distance ${s.method.intact_max_distance}, edited up to ${s.method.edited_max_distance}.`;
    }
  }

  async function main() {
    let summary = null, graves = [];
    try { summary = await getJSON("data/summary.json"); } catch (_) { summary = null; }
    renderMethod(summary);
    if (!summary || !summary.sample || !summary.sample.checked) { show("empty"); return; }
    try { graves = await getJSON("data/graveyard.json"); } catch (_) { graves = []; }
    renderHero(summary);
    renderYears(summary);
    renderSurvival(summary);
    let lastW = $("chart-survival").clientWidth, timer = null;
    window.addEventListener("resize", () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        const w = $("chart-survival").clientWidth;
        if (Math.abs(w - lastW) > 20) { lastW = w; renderSurvival(summary); }
      }, 150);
    });
    renderCauses(summary);
    renderTlds(summary);
    renderArchive(summary);
    renderGraveyard(graves, summary);
  }
  main();
})();
