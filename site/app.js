(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const show = (id) => { $(id).hidden = false; };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x, d = 0) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
  const num = (x) => (x == null ? "–" : x.toLocaleString("en-US"));
  const BUILD = new URL(document.currentScript.src).searchParams.get("v") || "dev";
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const FATES = {
    intact: ["alive", "same address, same words"],
    edited: ["alive", "same address, words revised"],
    moved: ["alive", "redirects to the same page elsewhere"],
    reachable: ["alive", "answers; front page or not comparable"],
    replaced: ["changed", "same address, different page"],
    redirect_home: ["dead", "deep link sent to the front page"],
    soft_404: ["dead", "answers 200, says not found"],
    not_found: ["dead", "404, 410, other 4xx"],
    parked: ["dead", "domain parked or for sale"],
    defunct: ["dead", "default server page, suspended, expired"],
    domain_gone: ["dead", "name no longer resolves"],
    blocked: ["unknown", "403, 429, bot wall"],
    server_error: ["unknown", "5xx"],
    unreachable: ["unknown", "timeout, refused, tls"],
  };
  const CAUSE = {
    not_found: "404", domain_gone: "no dns", parked: "parked", soft_404: "soft 404",
    redirect_home: "sent home", defunct: "defunct",
  };
  const GLYPH = { alive: "█", drift: "▒", archived: "░", lost: "·" };

  // ── tooltip ─────────────────────────
  const tip = $("tip");
  const showTip = (text, x, y) => {
    tip.textContent = text;
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    tip.style.left = `${Math.max(8, x + 14 + r.width > innerWidth ? x - r.width - 14 : x + 14)}px`;
    tip.style.top = `${Math.max(8, y + 14 + r.height > innerHeight ? y - r.height - 14 : y + 14)}px`;
  };
  const hideTip = () => { tip.hidden = true; };

  // Split `total` cells across shares so they sum exactly (largest remainder).
  function apportion(shares, total) {
    const sum = shares.reduce((a, b) => a + b, 0) || 1;
    const raw = shares.map((s) => (s / sum) * total);
    const out = raw.map(Math.floor);
    let left = total - out.reduce((a, b) => a + b, 0);
    raw.map((r, i) => [r - Math.floor(r), i]).sort((a, b) => b[0] - a[0]).forEach(([, i]) => { if (left-- > 0) out[i]++; });
    return out;
  }

  // ── big number in a block face (digits after ascii.rest's block font) ──
  const FACE = {
    0: "###|#.#|#.#|#.#|###", 1: ".#.|##.|.#.|.#.|###", 2: "###|..#|###|#..|###", 3: "###|..#|.##|..#|###",
    4: "#.#|#.#|###|..#|..#", 5: "###|#..|###|..#|###", 6: "###|#..|###|#.#|###", 7: "###|..#|..#|..#|..#",
    8: "###|#.#|###|#.#|###", 9: "###|#.#|###|..#|###", "%": "#...#|...#.|..#..|.#...|#...#", ".": ".|.|.|.|#",
  };
  function bigText(text) {
    const rows = Array.from({ length: 5 }, () => []);
    [...text].forEach((ch, k) => {
      const g = (FACE[ch] || FACE[0]).split("|");
      if (k) rows.forEach((r) => r.push(" ", " "));
      g.forEach((line, i) => [...line].forEach((p) => rows[i].push(p === "#" ? "█" : " ", p === "#" ? "█" : " ")));
    });
    const h = rows.length + 1, w = rows[0].length + 1;
    const grid = Array.from({ length: h }, (_, r) => Array.from({ length: w }, (_, c) => (rows[r] && rows[r][c]) || " "));
    for (let r = h - 2; r >= 0; r--) for (let c = w - 2; c >= 0; c--) {
      if (grid[r][c] === "█" && grid[r + 1][c + 1] === " ") grid[r + 1][c + 1] = "░";
    }
    return grid.map((r) => r.join("").replace(/\s+$/, "")).join("\n");
  }

  // ── hero: number and a field of 100 pages ───────────────────────────
  function renderHero(s) {
    const h = s.headline, b = s.totals.buckets;
    $("bignum").textContent = bigText(pct(h.dead_share));
    $("bignum").setAttribute("aria-label", `${pct(h.dead_share)} gone`);
    $("bigcap").textContent = `gone · n=${num(s.totals.known)}`;

    const arch = s.archive.dead_archived_share;
    const dead = b.dead;
    const counts = apportion([b.alive, b.drifted, arch == null ? 0 : dead * arch, arch == null ? dead : dead * (1 - arch)], 100);
    const target = [];
    ["alive", "drift", "archived", "lost"].forEach((k, i) => { for (let j = 0; j < counts[i]; j++) target.push(k); });
    const ladder = ["alive", "drift", "archived", "lost"];
    const cls = { alive: "", drift: "", archived: "g-dead", lost: "g-lost" };
    const field = $("field");
    field.setAttribute("aria-label", `of 100 pages: ${counts[0]} alive, ${counts[1]} changed, ${counts[2]} dead but archived, ${counts[3]} dead and lost`);
    const draw = (levels) => {
      let out = "";
      for (let i = 0; i < 100; i++) {
        const k = ladder[levels[i]];
        const cell = GLYPH[k] + GLYPH[k];
        out += cls[k] ? `<span class="${cls[k]}">${cell}</span>` : cell;
        out += i % 10 === 9 ? (i < 99 ? "\n" : "") : " ";
      }
      field.innerHTML = out;
    };
    const final = target.map((k) => ladder.indexOf(k));
    if (still) { draw(final); } else {
      const start = performance.now() + 300;
      const delay = final.map((_, i) => (99 - i) * 14 + Math.random() * 120);
      const tick = (now) => {
        let done = true;
        const levels = final.map((f, i) => {
          const steps = Math.floor(Math.max(0, now - start - delay[i]) / 90);
          if (steps < f) done = false;
          return Math.min(f, steps);
        });
        draw(levels);
        if (!done) requestAnimationFrame(tick);
      };
      draw(final.map(() => 0));
      requestAnimationFrame(tick);
    }
    $("key").innerHTML = [["alive", "alive"], ["drift", "changed"], ["archived", "dead, archived"], ["lost", "dead, lost"]]
      .map(([k, label]) => `<span><b>${GLYPH[k]}</b>${label}</span>`).join("");
    show("hero");
  }

  // ── departure board (custom ascii.rest piece) ───────────────────────
  function renderBoard(s, graves) {
    if (!graves.length) return;
    const rows = graves.slice().sort(() => Math.random() - 0.5).slice(0, 60)
      .map((g) => [(g.crawl_date || "").slice(0, 4), g.host || "", CAUSE[g.status] || g.status]);
    const el = document.createElement("ascii-art");
    el.setAttribute("src", `pieces/departed.js?v=${BUILD}`);
    el.setAttribute("label", "departures board of dead pages");
    el.setAttribute("options", JSON.stringify({ title: "departed", stamp: s.census_date, rows }));
    $("board").replaceChildren(el);
    show("board");
  }

  // ── strata: one row per capture year ────────────────────────────────
  function renderStrata(s) {
    const rows = s.by_year.filter((y) => y.known > 0);
    const draw = () => {
      const W = innerWidth < 600 ? 34 : innerWidth < 900 ? 48 : 64;
      $("strata-rows").innerHTML = rows.map((y) => {
        const [a, d, x] = apportion([y.buckets.alive, y.buckets.drifted, y.buckets.dead], W);
        return `<div class="stratum" data-y="${y.year}"><span class="y">${y.year}</span>` +
          `<span class="bar"><span class="g-alive">${GLYPH.alive.repeat(a)}</span><span class="g-drift">${GLYPH.drift.repeat(d)}</span><span class="g-dead">${GLYPH.lost.repeat(x)}</span></span>` +
          `<span class="p">${pct(y.dead_share)}</span></div>`;
      }).join("") +
        `<div class="strata-key"><span><b>${GLYPH.alive}</b>alive</span><span><b>${GLYPH.drift}</b>changed</span><span><b>${GLYPH.lost}</b>dead</span><span>right: dead share</span></div>`;
    };
    draw();
    const byYear = Object.fromEntries(rows.map((y) => [y.year, y]));
    const host = $("strata-rows");
    host.onmousemove = (e) => {
      const row = e.target.closest(".stratum");
      if (!row) return hideTip();
      const y = byYear[row.dataset.y];
      showTip(`${y.year}  n=${num(y.known)}\nalive    ${pct(y.alive_share, 1)}\nchanged  ${pct(y.buckets.drifted / y.known, 1)}\ndead     ${pct(y.dead_share, 1)}  [${pct(y.dead_ci[0])}–${pct(y.dead_ci[1])}]\nunknown  ${pct(y.unknown_share, 1)} of visits`, e.clientX, e.clientY);
    };
    host.onmouseleave = hideTip;
    let last = innerWidth;
    addEventListener("resize", () => { if (Math.abs(innerWidth - last) > 40) { last = innerWidth; draw(); } });
    show("strata");
  }

  // ── decay plot in characters ────────────────────────────────────────
  function renderDecay(s) {
    const pts = s.by_year.filter((y) => y.known >= 30 && y.survival != null);
    if (!pts.length) return;
    const surv = s.decay.survival, unch = s.decay.unchanged;
    const draw = () => {
      const W = innerWidth < 600 ? 40 : 66, H = 15, L = 5;
      const maxAge = Math.max(14, Math.ceil(Math.max(...pts.map((p) => p.age_years))));
      const grid = Array.from({ length: H + 2 }, () => Array.from({ length: L + W + 1 }, () => [" ", ""]));
      const put = (r, c, ch, cl) => { if (r >= 0 && r < grid.length && c >= 0 && c < grid[0].length) grid[r][c] = [ch, cl]; };
      const col = (t) => L + 1 + Math.round((t / maxAge) * (W - 1));
      const row = (v) => Math.round((1 - v) * (H - 1));
      for (let r = 0; r < H; r++) put(r, L, "│", "");
      [[1, "100"], [0.5, " 50"], [0, "  0"]].forEach(([v, lab]) => { [...lab].forEach((ch, i) => put(row(v), i, ch, "")); put(row(v), L, "┤", ""); });
      for (let c = L; c <= L + W; c++) put(H, c, "─", "");
      put(H, L, "└", "");
      for (let t = 0; t <= maxAge; t += 2) {
        const c = col(t);
        put(H, c, "┴", "");
        [...String(t)].forEach((ch, i) => put(H + 1, c + i - (t >= 10 ? 1 : 0), ch, ""));
      }
      const fit = (lam, ch) => { for (let c = L + 1; c <= L + W; c++) { const t = ((c - L - 1) / (W - 1)) * maxAge; put(row(Math.exp(-lam * t)), c, ch, "fit"); } };
      if (unch) fit(unch.lambda_per_year, "·");
      if (surv) fit(surv.lambda_per_year, "·");
      pts.forEach((p) => {
        if (p.unchanged != null) put(row(p.unchanged), col(p.age_years), "○", "pt");
        put(row(p.survival), col(p.age_years), "●", "pt");
      });
      if (surv && surv.half_life_years <= maxAge) {
        const c = col(surv.half_life_years);
        for (let r = row(0.5); r < H; r++) if (grid[r][c][0] === " ") put(r, c, "┊", "half");
        const lab = ` t½ ${surv.half_life_years.toFixed(1)}y`;
        const start = c + lab.length < L + W ? c : c - lab.length;
        [...lab].forEach((ch, i) => put(row(0.5) - 1, start + i, ch, "half"));
      }
      const lines = grid.map((r) => {
        let out = "", cur = null, buf = "";
        const flush = () => { if (buf) out += cur ? `<span class="${cur}">${esc(buf)}</span>` : esc(buf); buf = ""; };
        r.forEach(([ch, cl]) => { if (cl !== cur) { flush(); cur = cl; } buf += ch; });
        flush();
        return out.replace(/\s+$/, "");
      });
      lines.push("", `     ● still online   ○ still the same words   x: age in years   y: %`);
      $("plot").innerHTML = lines.join("\n");
      $("plot").setAttribute("aria-label", surv ? `half of pages gone after ${surv.half_life_years.toFixed(1)} years` : "decay plot");
    };
    draw();
    let last = innerWidth;
    addEventListener("resize", () => { if (Math.abs(innerWidth - last) > 40) { last = innerWidth; draw(); } });
    show("decay");
  }

  // ── bars ────────────────────────────────────────────────────────────
  function bars(el, items, width = 22) {
    const max = Math.max(...items.map((i) => i.value), 1e-9);
    const pad = Math.max(...items.map((i) => i.name.length));
    el.innerHTML = items.map((i) => {
      const n = Math.max(1, Math.round((i.value / max) * width));
      return `${esc(i.name.padEnd(pad))}  <span class="v">${"▇".repeat(n)}</span>${" ".repeat(width - n)}  ${pct(i.value).padStart(4)}${i.extra ? `  ${esc(i.extra)}` : ""}`;
    }).join("\n");
  }
  function renderCauses(s) {
    const items = Object.entries(s.dead_breakdown).map(([k, v]) => ({ name: CAUSE[k] || k, value: v })).sort((a, b) => b.value - a.value);
    if (!items.length) return;
    bars($("causes-rows"), items);
    show("causes");
  }
  function renderTlds(s) {
    if (!s.by_tld.length) return;
    bars($("tld-rows"), s.by_tld.map((t) => ({ name: `.${t.tld}`, value: t.dead_share, extra: `n=${num(t.known)}` })), 18);
    show("tlds");
  }

  // ── graves ──────────────────────────────────────────────────────────
  function stone(g) {
    const W = 11;
    let host = (g.host || "?").toLowerCase();
    if (host.length > W) host = host.slice(0, W - 2) + "..";
    const center = (t) => { const l = Math.floor((W - t.length) / 2); return " ".repeat(l) + t + " ".repeat(W - t.length - l); };
    const year = (g.crawl_date || "").slice(0, 4);
    const art = [
      "    _______    ",
      "  .'       '.  ",
      ` /${center("r.i.p")}\\ `,
      ` |${" ".repeat(W)}| `,
      ` |<span class="h">${esc(center(host))}</span>| `,
      ` |${center(year)}| `,
      ` |${" ".repeat(W)}| `,
      `_|${"_".repeat(W)}|_`,
    ].join("\n");
    return `<a class="stone" href="${esc(g.wayback)}" target="_blank" rel="noopener nofollow" title="${esc(g.url)}">` +
      `<pre>${art}</pre><span class="s">${esc(CAUSE[g.status] || g.status)}${g.archived === true ? " · archived" : g.archived === false ? " · lost" : ""} ↗</span></a>`;
  }
  function renderGraves(graves) {
    if (!graves.length) return;
    const draw = () => {
      const n = innerWidth < 600 ? 6 : 12;
      const pool = graves.slice();
      const pick = [];
      while (pick.length < n && pool.length) pick.push(pool.splice(Math.floor(Math.random() * pool.length), 1)[0]);
      $("stones").innerHTML = pick.map(stone).join("");
    };
    $("shuffle").onclick = draw;
    draw();
    show("graves");
  }

  // ── method ──────────────────────────────────────────────────────────
  function renderMethod(s) {
    $("fates").innerHTML = Object.entries(FATES).map(([k, [b, t]]) => `<tr><td>${k.replace("_", " ")}</td><td>${b}</td><td>${esc(t)}</td></tr>`).join("");
    if (!s) return;
    const m = s.method, d = s.decay;
    $("method-facts").textContent = [
      `sample     ${num(s.sample.urls)} urls · ${s.sample.crawls} crawls · ${num(s.sample.hosts)} sites visited`,
      `           ${(s.sample.first_crawl_date || "").slice(0, 7)} to ${(s.sample.last_crawl_date || "").slice(0, 7)} · up to 1,000 per crawl, fixed by url hash`,
      `visit      1 get per url · 2 per host · 15 s timeout · 512 kb cap · monthly`,
      `compare    share of the original 3-word shingles still on the page (32-hash sketch)`,
      `           intact ≥ ${m.intact_min_retained} · edited ≥ ${m.edited_min_retained} · a front page that answers is alive`,
      `half-life  ${d.survival ? d.survival.half_life_years.toFixed(1) + " y (pages)" : "–"} · ${d.unchanged ? d.unchanged.half_life_years.toFixed(1) + " y (words)" : "–"}`,
      `archive    ${num(s.archive.dead_checked)} dead pages checked · ${pct(s.archive.dead_archived_share)} have a wayback copy`,
      `unknown    ${pct(s.headline.unknown_share)} of visits; left out of every share`,
    ].join("\n");
    $("year-table").innerHTML = `<table><thead><tr><th>year</th><th>n</th><th>alive</th><th>changed</th><th>dead</th><th>95% ci</th><th>unknown</th></tr></thead><tbody>` +
      s.by_year.filter((y) => y.known).map((y) => `<tr><td>${y.year}</td><td>${num(y.known)}</td><td>${pct(y.alive_share, 1)}</td><td>${pct(y.buckets.drifted / y.known, 1)}</td><td>${pct(y.dead_share, 1)}</td><td>${pct(y.dead_ci[0])}–${pct(y.dead_ci[1])}</td><td>${pct(y.unknown_share, 1)}</td></tr>`).join("") +
      `</tbody></table>`;
  }

  async function getJSON(path) {
    const r = await fetch(path, { cache: "no-cache" });
    if (!r.ok) throw new Error(path);
    return r.json();
  }

  async function main() {
    let s = null, graves = [];
    try { s = await getJSON("data/summary.json"); } catch (_) { s = null; }
    renderMethod(s);
    if (!s || !s.sample || !s.sample.checked) { show("nosignal"); return; }
    try { graves = await getJSON("data/graveyard.json"); } catch (_) { graves = []; }
    $("meta").textContent = `census ${s.census_date} · ${num(s.sample.checked)} pages`;
    renderHero(s);
    renderBoard(s, graves);
    renderStrata(s);
    renderDecay(s);
    renderCauses(s);
    renderTlds(s);
    renderGraves(graves);
  }
  main();
})();
