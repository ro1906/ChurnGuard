"""The Overview hero: a live leaky bucket and a "win them back" simulator.

Customers are dots in a bucket of recurring revenue. High-risk dots drift towards the leaks and
fall out. Moving the slider saves a share of them (they turn green and stay), so the viewer
sees what a retention programme is worth. It runs in an iframe so it can use a canvas and its
own script; everything it shows is computed from the deployed model's scores.
"""

from __future__ import annotations

import json

from churnguard.ui.theme import font_data_uri

_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
@font-face { font-family: "Bricolage Grotesque"; font-weight: 200 800;
  src: url(__BRICOLAGE__) format("woff2"); }
@font-face { font-family: "Figtree"; font-weight: 300 900; src: url(__FIGTREE__) format("woff2"); }
:root { --ink:#0C2E3B; --paper:#EAF2EE; --card:#FBFDFC; --water:#12788A; --coral:#F2574B;
  --amber:#F5B83D; --kelp:#2FA877; --muted:#4F6B73; }
* { box-sizing: border-box; }
html, body { margin: 0; background: transparent; color: var(--ink);
  font-family: Figtree, system-ui, sans-serif; -webkit-font-smoothing: antialiased; }
.wrap { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 32px;
  align-items: center; max-width: 1180px; margin: 0 auto; padding: 8px 4px 16px; }
h1 { font-family: "Bricolage Grotesque", sans-serif; font-weight: 800; letter-spacing: -.035em;
  font-size: clamp(2.7rem, 5.6vw, 4.6rem); line-height: .96; margin: 0 0 18px; }
.lede { font-size: 1.15rem; line-height: 1.55; color: var(--muted); max-width: 34rem; margin: 0 0 26px; }
.stats { display: flex; gap: 34px; margin-bottom: 26px; flex-wrap: wrap; }
.stat b { display: block; font-family: "Bricolage Grotesque", sans-serif; font-weight: 800;
  font-size: clamp(2rem, 3.6vw, 2.9rem); letter-spacing: -.04em; line-height: 1; color: var(--coral); }
.stat span { display: block; margin-top: 6px; color: var(--muted); font-size: .98rem; max-width: 13rem; line-height: 1.3; }
.sim { background: var(--card); border-radius: 26px; padding: 22px 24px 20px;
  box-shadow: 0 24px 40px -30px rgba(12,46,59,.55); border: 1px solid rgba(12,46,59,.07); max-width: 33rem; }
.sim h2 { font-family: "Bricolage Grotesque", sans-serif; font-size: 1.25rem; font-weight: 800;
  letter-spacing: -.02em; margin: 0 0 14px; }
.ctl { display: grid; grid-template-columns: 1fr auto; gap: 4px 12px; align-items: baseline; margin-bottom: 12px; }
.ctl label { font-weight: 600; font-size: .98rem; }
.ctl output { font-family: "Bricolage Grotesque", sans-serif; font-weight: 800; font-size: 1.25rem; }
.ctl input { grid-column: 1 / -1; }
input[type=range] { -webkit-appearance: none; appearance: none; width: 100%; height: 10px; border-radius: 99px;
  background: linear-gradient(90deg, var(--kelp) var(--p, 25%), rgba(12,46,59,.12) var(--p, 25%)); outline: none; margin: 8px 0 2px; }
input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 26px; height: 26px; border-radius: 50%;
  background: #fff; border: 4px solid var(--ink); cursor: grab; box-shadow: 0 4px 10px rgba(12,46,59,.3); }
input[type=range]::-moz-range-thumb { width: 20px; height: 20px; border-radius: 50%; background: #fff;
  border: 4px solid var(--ink); cursor: grab; }
input[type=range]:focus-visible { outline: 3px solid #7FD1D9; outline-offset: 4px; }
.out { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 16px;
  border-top: 1px solid rgba(12,46,59,.1); padding-top: 16px; }
.out b { display: block; font-family: "Bricolage Grotesque", sans-serif; font-weight: 800;
  font-size: clamp(1.25rem, 2.2vw, 1.7rem); letter-spacing: -.03em; color: var(--kelp); line-height: 1.05; }
.out span { font-size: .86rem; color: var(--muted); line-height: 1.25; display: block; margin-top: 4px; }
.fine { font-size: .82rem; color: var(--muted); margin: 12px 0 0; line-height: 1.4; }
.stage { position: relative; }
canvas { width: 100%; height: auto; aspect-ratio: 400 / 470; display: block; }
.tip { position: absolute; left: 50%; bottom: -2%; transform: translateX(-50%); font-size: .86rem; color: var(--muted);
  background: rgba(251,253,252,.85); padding: 4px 12px; border-radius: 99px; white-space: nowrap; }
@media (max-width: 860px) {
  .wrap { grid-template-columns: 1fr; gap: 12px; }
  .stage { order: -1; max-width: 420px; margin: 0 auto; width: 100%; }
}
</style></head>
<body>
<div class="wrap">
  <div>
    <h1>Stop the leak before customers walk.</h1>
    <p class="lede">ChurnGuard predicts which customers are about to leave, explains why, and shows the
      retention move most likely to keep them.</p>
    <div class="stats">
      <div class="stat"><b id="sHigh">0</b><span>customers likely to leave soon</span></div>
      <div class="stat"><b id="sRev">0</b><span>in monthly revenue riding on them</span></div>
    </div>
    <div class="sim">
      <h2>What if we win some of them back?</h2>
      <div class="ctl">
        <label for="save">Share of at-risk customers we keep</label><output id="saveOut">25%</output>
        <input id="save" type="range" min="0" max="100" step="5" value="25">
      </div>
      <div class="ctl">
        <label for="cost">Cost of each retention offer</label><output id="costOut">__CUR__20</output>
        <input id="cost" type="range" min="0" max="100" step="5" value="20">
      </div>
      <div class="out">
        <div><b id="oKept">0</b><span>customers kept</span></div>
        <div><b id="oYear">0</b><span>revenue kept over 12 months</span></div>
        <div><b id="oNet">0</b><span>left after paying for offers</span></div>
      </div>
      <p class="fine">An illustration built from the model's scores. Real save rates come from testing offers on a sample.</p>
    </div>
  </div>
  <div class="stage">
    <canvas id="c" width="400" height="470" role="img"
      aria-label="A bucket of customers. At-risk customers leak out through holes; moving the slider plugs the holes."></canvas>
    <div class="tip">Coral dots are customers likely to leave</div>
  </div>
</div>
<script>
const D = __DATA__;
const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const money = n => D.currency + Math.round(n).toLocaleString("en-US");
const compact = n => n >= 1e6 ? D.currency + (n / 1e6).toFixed(2) + "M" : n >= 1e4 ? D.currency + Math.round(n / 1e3) + "k" : money(n);
const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const rnd = (a, b) => a + Math.random() * (b - a);

/* ---------- headline numbers count up once ---------- */
function countUp(el, to, fmt) {
  if (reduce) { el.textContent = fmt(to); return; }
  const t0 = performance.now(), dur = 1400;
  (function step(t) {
    const k = clamp((t - t0) / dur, 0, 1), e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(to * e);
    if (k < 1) requestAnimationFrame(step);
  })(t0);
}
countUp($("sHigh"), D.high, n => Math.round(n).toLocaleString("en-US"));
countUp($("sRev"), D.high_revenue, money);

/* ---------- simulator ---------- */
let save = 0.25, cost = 20;
function paintRange(el) { el.style.setProperty("--p", ((el.value - el.min) / (el.max - el.min) * 100) + "%"); }
function recalc() {
  save = +$("save").value / 100; cost = +$("cost").value;
  $("saveOut").textContent = Math.round(save * 100) + "%";
  $("costOut").textContent = D.currency + cost;
  const kept = Math.round(D.high * save), year = kept * D.avg_high_charge * 12, net = year - D.high * cost;
  $("oKept").textContent = kept.toLocaleString("en-US");
  $("oYear").textContent = compact(year);
  $("oNet").textContent = (net < 0 ? "-" : "") + compact(Math.abs(net));
  $("oNet").style.color = net < 0 ? "var(--coral)" : "var(--kelp)";
  paintRange($("save")); paintRange($("cost"));
}
["save", "cost"].forEach(id => $(id).addEventListener("input", recalc));
$("cost").style.setProperty("--p", "20%");
recalc();

/* ---------- the bucket ---------- */
const cv = $("c"), ctx = cv.getContext("2d");
const W = 400, H = 470, GROUND = 432, WATER_TOP = 158, FLOOR = 370;
const L0 = 56, L1 = 90, R0 = 344, R1 = 310, TOP = 88, BOT = 382;
const leftX = y => L0 + (L1 - L0) * (y - TOP) / (BOT - TOP);
const rightX = y => R0 + (R1 - R0) * (y - TOP) / (BOT - TOP);
const holes = [
  { y: 252, side: -1 }, { y: 320, side: -1 }, { y: 286, side: 1 },
].map(h => ({ ...h, x: h.side < 0 ? leftX(h.y) : rightX(h.y) }));

function fit() {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.width * dpr * H / W);
  ctx.setTransform(cv.width / W, 0, 0, cv.width / W, 0, 0);
}
window.addEventListener("resize", fit); fit();

const total = D.high + D.medium + D.low;
const N = 66, nHigh = Math.max(9, Math.round(N * D.high / total)), nMed = Math.round(N * D.medium / total);
const dots = [], splashes = [], respawn = [];
function place(d, type) {
  d.type = type; d.state = "in"; d.r = rnd(7, 9.5); d.green = 0; d.savedAt = 0; d.decided = false; d.hole = null;
  d.y = rnd(WATER_TOP + 24, FLOOR - 14); d.x = rnd(leftX(d.y) + 20, rightX(d.y) - 20);
  d.vx = rnd(-14, 14); d.vy = rnd(-14, 14); return d;
}
for (let i = 0; i < N; i++) dots.push(place({}, i < nHigh ? "high" : i < nHigh + nMed ? "med" : "low"));

function nearestHole(d) {
  let best = holes[0], bd = 1e9;
  for (const h of holes) { const dd = Math.hypot(h.x - d.x, h.y - d.y) + (h.taken || 0); if (dd < bd) { bd = dd; best = h; } }
  return best;
}

function update(dt) {
  for (const d of dots) {
    if (d.state === "in") {
      d.vx += rnd(-1, 1) * 60 * dt; d.vy += rnd(-1, 1) * 60 * dt;
      if (d.type === "high" && d.green === 0) {
        d.hole = d.hole || nearestHole(d);
        const tx = d.hole.x - d.hole.side * 16, ty = d.hole.y, dx = tx - d.x, dy = ty - d.y, dist = Math.hypot(dx, dy) || 1;
        d.vx += dx / dist * 70 * dt; d.vy += dy / dist * 70 * dt;
        if (dist < 14) {
          if (Math.random() < save) { const side = d.hole.side; d.green = 1; d.savedAt = performance.now(); d.hole = null; d.vx = -side * 40; d.vy = rnd(-30, 30); }
          else { d.state = "fall"; d.vx = d.hole.side * 70; d.vy = -20; d.x = d.hole.x + d.hole.side * 6; d.y = d.hole.y; continue; }
        }
      }
      if (d.green && performance.now() - d.savedAt > 6000) d.green = 0, d.type = "low";
      const sp = Math.hypot(d.vx, d.vy), cap = d.type === "high" ? 70 : 34;
      if (sp > cap) { d.vx *= cap / sp; d.vy *= cap / sp; }
      d.x += d.vx * dt; d.y += d.vy * dt;
      const lo = leftX(d.y) + d.r + 5, hi = rightX(d.y) - d.r - 5, top = WATER_TOP + d.r + 10, bot = FLOOR - d.r;
      if (d.x < lo) { d.x = lo; d.vx = Math.abs(d.vx); } if (d.x > hi) { d.x = hi; d.vx = -Math.abs(d.vx); }
      if (d.y < top) { d.y = top; d.vy = Math.abs(d.vy); } if (d.y > bot) { d.y = bot; d.vy = -Math.abs(d.vy); }
    } else if (d.state === "fall") {
      d.vy += 760 * dt; d.x += d.vx * dt; d.y += d.vy * dt; d.vx *= 0.985;
      if (d.y >= GROUND) { splashes.push({ x: d.x, t: 0 }); d.state = "gone"; respawn.push({ d, at: performance.now() + rnd(900, 2200) }); }
    }
  }
  const now = performance.now();
  for (let i = respawn.length - 1; i >= 0; i--) if (now >= respawn[i].at) { place(respawn[i].d, "high"); respawn.splice(i, 1); }
  for (let i = splashes.length - 1; i >= 0; i--) { splashes[i].t += dt; if (splashes[i].t > 0.6) splashes.splice(i, 1); }
}

function bucketPath(c) {
  c.beginPath(); c.moveTo(L0, TOP); c.lineTo(R0, TOP); c.lineTo(R1, BOT - 14); c.quadraticCurveTo(R1 - 2, BOT, R1 - 18, BOT);
  c.lineTo(L1 + 18, BOT); c.quadraticCurveTo(L1 + 2, BOT, L1, BOT - 14); c.closePath();
}
function dotColor(d) {
  if (d.green) return "#35C189";
  return d.type === "high" ? "#F2574B" : d.type === "med" ? "#F5B83D" : "rgba(255,255,255,.88)";
}

function draw(t) {
  ctx.clearRect(0, 0, W, H);
  // ground shadow + puddle
  ctx.fillStyle = "rgba(12,46,59,.10)"; ctx.beginPath(); ctx.ellipse(200, GROUND + 6, 150, 12, 0, 0, 7); ctx.fill();
  // bail (handle)
  ctx.strokeStyle = "#0C2E3B"; ctx.lineWidth = 5; ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(L0 + 10, TOP + 4); ctx.quadraticCurveTo(200, -52, R0 - 10, TOP + 4); ctx.stroke();
  // glass
  ctx.save(); bucketPath(ctx); ctx.fillStyle = "rgba(255,255,255,.55)"; ctx.fill(); ctx.clip();
  // water with a gentle wave
  const g = ctx.createLinearGradient(0, WATER_TOP, 0, BOT); g.addColorStop(0, "#58BBCB"); g.addColorStop(1, "#12788A");
  ctx.fillStyle = g; ctx.beginPath(); ctx.moveTo(40, BOT + 10);
  for (let x = 40; x <= 360; x += 6) ctx.lineTo(x, WATER_TOP + Math.sin(x / 26 + t / 700) * 4 + Math.sin(x / 11 - t / 900) * 1.6);
  ctx.lineTo(360, BOT + 10); ctx.closePath(); ctx.fill();
  // customers
  for (const d of dots) if (d.state === "in") {
    ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, 7); ctx.fillStyle = dotColor(d); ctx.fill();
    if (d.green) { ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = 2; ctx.stroke(); }
  }
  ctx.restore();
  // outline and rim
  ctx.strokeStyle = "#0C2E3B"; ctx.lineWidth = 7; ctx.lineJoin = "round"; bucketPath(ctx); ctx.stroke();
  ctx.fillStyle = "#0C2E3B"; ctx.beginPath(); ctx.roundRect(L0 - 12, TOP - 12, R0 - L0 + 24, 22, 11); ctx.fill();
  // holes, with a green seal that grows as more customers are saved
  for (const h of holes) {
    const r = 11 - 6 * save;
    ctx.fillStyle = "#0C2E3B"; ctx.beginPath(); ctx.arc(h.x, h.y, r + 3, 0, 7); ctx.fill();
    ctx.fillStyle = "#EAF2EE"; ctx.beginPath(); ctx.arc(h.x, h.y, r, 0, 7); ctx.fill();
    if (save > 0) { ctx.strokeStyle = "#35C189"; ctx.lineWidth = 4; ctx.beginPath(); ctx.arc(h.x, h.y, r + 7, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * save); ctx.stroke(); }
  }
  // falling customers and splashes
  for (const d of dots) if (d.state === "fall") { ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, 7); ctx.fillStyle = "#F2574B"; ctx.fill(); }
  for (const s of splashes) {
    ctx.strokeStyle = `rgba(242,87,75,${1 - s.t / 0.6})`; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.ellipse(s.x, GROUND + 4, 8 + s.t * 60, 3 + s.t * 10, 0, 0, 7); ctx.stroke();
  }
}

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  if (!document.hidden) { update(dt); draw(now); }
  requestAnimationFrame(frame);
}
if (reduce) { for (let i = 0; i < 30; i++) update(0.05); draw(0); ["save", "cost"].forEach(id => $(id).addEventListener("input", () => draw(0))); }
else requestAnimationFrame(frame);
</script></body></html>
"""


def hero_html(stats: dict, currency: str) -> str:
    """Return the hero page as an HTML string for st.iframe.

    `stats` holds plain numbers only (counts and revenue), so embedding them as JSON is safe.
    """
    data = {**stats, "currency": currency}
    return (
        _TEMPLATE.replace("__BRICOLAGE__", font_data_uri("bricolage"))
        .replace("__FIGTREE__", font_data_uri("figtree"))
        .replace("__DATA__", json.dumps(data))
        .replace("__CUR__", currency)
    )
