/*
 * departed: a split-flap board of web pages that no longer exist.
 * An ascii.rest piece (https://ascii.rest, MIT), adapted from its split-flap board.
 *
 * options.rows: [[year, site, cause], ...]
 */

export const meta = {
  name: "departed",
  category: "type",
  note: "a departures board of dead web pages, flaps riffling to each new line",
  cols: 62,
  rows: 18,
  fps: 15,
  options: {
    title: "departed",
    stamp: "",
    rows: [["2014", "example.com", "404"]],
  },
};

const TEXT = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-/'&";
const STEP = 3.2;
const LEAD = 0.8;
const RATE = 30;
const SHOWN = 6;

const hash = (a, b, c) => {
  let h = Math.imul(a, 374761393) + Math.imul(b, 668265263) + Math.imul(c, 2246822519);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
};
const clean = (s) => [...String(s ?? "").toUpperCase()].map((ch) => (TEXT.includes(ch) ? ch : " ")).join("");

export default function departed({ title = meta.options.title, stamp = meta.options.stamp, rows: list = meta.options.rows } = {}) {
  const { cols, rows } = meta;
  const entries = (Array.isArray(list) && list.length ? list : meta.options.rows).map((r) => r.map(clean));
  const n = entries.length;
  const at = (s, i) => entries[(((Math.floor(s / 1) + i) % n) + n) % n];

  // [row, first column, tiles, text on board s]
  const fields = [];
  for (let i = 0; i < SHOWN; i++) {
    const r = 5 + i * 2;
    const live = () => i < n;
    fields.push(
      [r, 2, 4, (s) => (live() ? at(s, i)[0] : "")],
      [r, 12, 14, (s) => (live() ? at(s, i)[1] : "")],
      [r, 42, 9, (s) => (live() ? at(s, i)[2] : "")],
    );
  }

  const base = [];
  for (let r = 0; r < rows; r++) base.push(("│" + " ".repeat(cols - 2) + "│").split(""));
  base[0] = ("╭" + "─".repeat(cols - 2) + "╮").split("");
  base[3] = ("├" + "─".repeat(cols - 2) + "┤").split("");
  base[rows - 1] = ("╰" + "─".repeat(cols - 2) + "╯").split("");
  const put = (r, c, s) => [...s].forEach((ch, i) => c + i < cols - 1 && (base[r][c + i] = ch));
  put(1, 2, [...clean(title).slice(0, 14)].join(" "));
  if (stamp) put(1, cols - 2 - String(stamp).length, String(stamp));
  put(4, 2, "captured");
  put(4, 12, "site");
  put(4, 42, "cause");
  for (const [r, c, len] of fields) for (let i = 0; i < len; i++) base[r + 1][c + 2 * i] = "‾";

  return (t) => {
    const v = t - LEAD + STEP;
    const k = Math.floor(v / STEP);
    const since = v - k * STEP;
    const s = k;
    const grid = base.map((row) => row.slice());
    for (const [r, c, len, text] of fields) {
      const to = text(s).padEnd(len).slice(0, len);
      const from = text(s - 1).padEnd(len).slice(0, len);
      for (let i = 0; i < len; i++) {
        const a = Math.max(0, TEXT.indexOf(from[i])), b = Math.max(0, TEXT.indexOf(to[i]));
        let ch = TEXT[b];
        if (a !== b) {
          const begin = r * 0.02 + (c + 2 * i) * 0.004 + hash(r, c + i, s) * 0.12;
          const flips = Math.floor(Math.max(0, since - begin) * RATE);
          const need = (b - a + TEXT.length) % TEXT.length;
          if (flips < need) {
            ch = TEXT[(a + flips) % TEXT.length];
            if (since >= begin) grid[r + 1][c + 2 * i] = "▀";
          }
        }
        grid[r][c + 2 * i] = ch;
      }
    }
    return grid.map((row) => row.join("")).join("\n");
  };
}
