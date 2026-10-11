/* Gráficas SVG mínimas para el reporte: líneas (ejes lin/log, bandas, crosshair), dispersión y
   trayectorias x-y. Sin dependencias. Colores vía tokens CSS (--s1.. , --ink, --muted, --grid). */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) {
      if ((k === "stroke" || k === "fill") && String(attrs[k]).startsWith("var(")) e.style.setProperty(k, attrs[k]);
      else e.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(e);
    return e;
  };
  const sup = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const pow10 = (e) => "10" + String(e).split("").map((c) => sup[c]).join("");
  const fmt = (v) => {
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a >= 1e4 || a < 1e-2) {
      const e = Math.floor(Math.log10(a)), m = v / Math.pow(10, e);
      return m.toFixed(1) + "·" + pow10(e);
    }
    return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toFixed(3);
  };
  function scale(lo, hi, a, b, log) {
    if (log) { lo = Math.log10(lo); hi = Math.log10(hi); }
    const f = (v) => a + ((log ? Math.log10(v) : v) - lo) / (hi - lo) * (b - a);
    f.inv = (p) => { const u = lo + (p - a) / (b - a) * (hi - lo); return log ? Math.pow(10, u) : u; };
    return f;
  }
  function ticks(lo, hi, log, n) {
    if (log) {
      const out = []; const a = Math.ceil(Math.log10(lo) - 1e-9), b = Math.floor(Math.log10(hi) + 1e-9);
      const st = Math.max(1, Math.ceil((b - a + 1) / (n || 6)));
      for (let e = a; e <= b; e += st) out.push([Math.pow(10, e), pow10(e)]);
      return out;
    }
    const span = hi - lo, raw = span / (n || 5), mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= (n || 5) + 1);
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * span; v += step) {
      const r = Math.abs(v) < step * 1e-9 ? 0 : v;
      out.push([r, String(+r.toPrecision(6))]);
    }
    return out;
  }
  function extent(arrs, log) {
    let lo = Infinity, hi = -Infinity;
    for (const a of arrs) for (const v of a) if (isFinite(v) && (!log || v > 0)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    return [lo, hi];
  }
  function frame(host, o, W, H, m) {
    host.classList.add("chart");
    if (o.title) { const h = document.createElement("div"); h.className = "chart-title"; h.textContent = o.title; host.appendChild(h); }
    if (o.series && o.series.length > 1 && o.legend !== false) {
      const lg = document.createElement("div"); lg.className = "legend";
      const seen = new Set();
      for (const s of o.series) {
        if (!s.name || seen.has(s.name)) continue; seen.add(s.name);
        const i = document.createElement("span"); i.className = "lg-item";
        i.innerHTML = `<svg width="18" height="10" aria-hidden="true"><line x1="1" y1="5" x2="17" y2="5" stroke="${s.color}" stroke-width="2.5" ${s.dash ? 'stroke-dasharray="' + s.dash + '"' : ""} stroke-linecap="round"/></svg>`;
        i.appendChild(document.createTextNode(s.name)); lg.appendChild(i);
      }
      host.appendChild(lg);
    }
    const wrap = document.createElement("div"); wrap.className = "plot"; host.appendChild(wrap);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title || o.aria || "gráfica" }, null);
    wrap.appendChild(svg);
    const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; wrap.appendChild(tip);
    return { svg, tip, wrap };
  }
  function axes(svg, o, X, Y, xr, yr, W, H, m) {
    const g = el("g", { class: "axes" }, svg);
    for (const [v, lab] of ticks(yr[0], yr[1], o.ylog, o.yticks)) {
      const y = Y(v);
      el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "grid" }, g);
      el("text", { x: m.l - 7, y: y + 4, class: "tick", "text-anchor": "end" }, g).textContent = lab;
    }
    for (const [v, lab] of ticks(xr[0], xr[1], o.xlog, o.xticks)) {
      const x = X(v);
      el("line", { x1: x, x2: x, y1: H - m.b, y2: H - m.b + 4, class: "axis" }, g);
      el("text", { x: x, y: H - m.b + 17, class: "tick", "text-anchor": "middle" }, g).textContent = lab;
    }
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, g);
    if (o.xlabel) el("text", { x: (m.l + W - m.r) / 2, y: H - 4, class: "alabel", "text-anchor": "middle" }, g).textContent = o.xlabel;
    if (o.ylabel) el("text", { x: 12, y: (m.t + H - m.b) / 2, class: "alabel", "text-anchor": "middle", transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` }, g).textContent = o.ylabel;
  }

  window.lineChart = function (host, o) {
    if (typeof host === "string") host = document.getElementById(host);
    const W = o.w || 540, H = o.h || 330, m = { l: 56, r: 14, t: 10, b: o.xlabel ? 46 : 28 };
    const { svg, tip, wrap } = frame(host, o, W, H, m);
    const xs = o.series.map((s) => s.x), ys = o.series.flatMap((s) => (s.band ? [s.y, s.band[0], s.band[1]] : [s.y]));
    let xr = o.xr || extent(xs, o.xlog), yr = o.yr || extent(ys, o.ylog);
    if (!o.yr) {
      if (o.ylog) yr = [Math.pow(10, Math.floor(Math.log10(yr[0]) - 0.05)), Math.pow(10, Math.ceil(Math.log10(yr[1]) + 0.05))];
      else { const p = (yr[1] - yr[0]) * 0.06 || 1; yr = [yr[0] - p, yr[1] + p]; }
    }
    const X = scale(xr[0], xr[1], m.l, W - m.r, o.xlog), Y = scale(yr[0], yr[1], H - m.b, m.t, o.ylog);
    axes(svg, o, X, Y, xr, yr, W, H, m);
    const ok = (x, y) => isFinite(x) && isFinite(y) && (!o.xlog || x > 0) && (!o.ylog || y > 0);
    const cl = (y) => Math.max(m.t - 40, Math.min(H - m.b + 40, y));
    const clipId = "c" + Math.random().toString(36).slice(2);
    el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }, el("clipPath", { id: clipId }, el("defs", {}, svg)));
    const body = el("g", { "clip-path": `url(#${clipId})` }, svg);
    for (const s of o.series) {
      if (s.band) {
        let d = "", back = "";
        s.x.forEach((x, i) => { if (ok(x, s.band[0][i]) && ok(x, s.band[1][i])) { d += (d ? "L" : "M") + X(x).toFixed(1) + " " + cl(Y(s.band[1][i])).toFixed(1); back = "L" + X(x).toFixed(1) + " " + cl(Y(s.band[0][i])).toFixed(1) + back; } });
        if (d) el("path", { d: d + back + "Z", fill: s.color, "fill-opacity": 0.14, stroke: "none" }, body);
      }
      let d = "", pen = false;
      s.x.forEach((x, i) => { if (ok(x, s.y[i])) { d += (pen ? "L" : "M") + X(x).toFixed(1) + " " + cl(Y(s.y[i])).toFixed(1); pen = true; } else pen = false; });
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dash || "none" }, body);
      if (s.markers) s.x.forEach((x, i) => { if (ok(x, s.y[i])) el("circle", { cx: X(x), cy: Y(s.y[i]), r: 3.6, fill: s.color, stroke: "var(--bg)", "stroke-width": 1.5 }, body); });
    }
    if (o.hline != null) { el("line", { x1: m.l, x2: W - m.r, y1: Y(o.hline), y2: Y(o.hline), class: "ref" }, body); }
    // crosshair + tooltip
    const cross = el("line", { y1: m.t, y2: H - m.b, class: "cross", visibility: "hidden" }, svg);
    const dots = o.series.map((s) => el("circle", { r: 4, fill: s.color, stroke: "var(--bg)", "stroke-width": 1.5, visibility: "hidden" }, svg));
    const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    const move = (ev) => {
      const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W;
      const xv = X.inv(Math.max(m.l, Math.min(W - m.r, px)));
      const base = o.series[0].x; let bi = 0, bd = Infinity;
      const dist = (a) => (o.xlog ? Math.abs(Math.log(a / xv)) : Math.abs(a - xv));
      base.forEach((a, i) => { const d = dist(a); if (d < bd) { bd = d; bi = i; } });
      const xx = base[bi];
      cross.setAttribute("x1", X(xx)); cross.setAttribute("x2", X(xx)); cross.setAttribute("visibility", "visible");
      let html = `<div class="tip-h">${o.xname || o.xlabel || "x"} = ${o.xfmt ? o.xfmt(xx) : fmt(xx)}</div>`;
      o.series.forEach((s, k) => {
        let j = 0, dd = Infinity; s.x.forEach((a, i) => { const d = o.xlog ? Math.abs(Math.log(a / xx)) : Math.abs(a - xx); if (d < dd) { dd = d; j = i; } });
        if (ok(s.x[j], s.y[j]) && s.name) {
          dots[k].setAttribute("cx", X(s.x[j])); dots[k].setAttribute("cy", cl(Y(s.y[j]))); dots[k].setAttribute("visibility", "visible");
          html += `<div class="tip-r"><i style="background:${s.color}"></i><span>${s.name}</span><b>${fmt(s.y[j])}</b></div>`;
        } else dots[k].setAttribute("visibility", "hidden");
      });
      tip.innerHTML = html; tip.hidden = false;
      const wr = wrap.getBoundingClientRect(), tx = ev.clientX - wr.left;
      tip.style.left = Math.min(Math.max(4, tx + (tx > wr.width * 0.6 ? -tip.offsetWidth - 12 : 12)), wr.width - tip.offsetWidth - 4) + "px";
      tip.style.top = Math.max(4, ev.clientY - wr.top - tip.offsetHeight - 8) + "px";
    };
    const leave = () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); };
    hit.addEventListener("pointermove", move); hit.addEventListener("pointerdown", move); hit.addEventListener("pointerleave", leave);
  };

  window.scatterChart = function (host, o) {
    if (typeof host === "string") host = document.getElementById(host);
    const W = o.w || 420, H = o.h || 380, m = { l: 58, r: 14, t: 10, b: 44 };
    const { svg, tip, wrap } = frame(host, o, W, H, m);
    const all = o.groups.flatMap((g) => g.x.concat(g.y));
    let [lo, hi] = extent([all], true);
    lo = Math.pow(10, Math.floor(Math.log10(lo))); hi = Math.pow(10, Math.ceil(Math.log10(hi)));
    const X = scale(lo, hi, m.l, W - m.r, true), Y = scale(lo, hi, H - m.b, m.t, true);
    o.xlog = o.ylog = true;
    axes(svg, o, X, Y, [lo, hi], [lo, hi], W, H, m);
    el("line", { x1: X(lo), y1: Y(lo), x2: X(hi), y2: Y(hi), class: "ref" }, svg);
    for (const g of o.groups) g.x.forEach((x, i) => {
      if (!(x > 0 && g.y[i] > 0)) return;
      const c = el("circle", { cx: X(x), cy: Y(g.y[i]), r: 2.6, fill: g.color, "fill-opacity": 0.55 }, svg);
      c.addEventListener("pointerenter", (ev) => {
        tip.innerHTML = `<div class="tip-h">${g.name}</div><div class="tip-r"><span>${o.xname}</span><b>${fmt(x)}</b></div><div class="tip-r"><span>${o.yname}</span><b>${fmt(g.y[i])}</b></div>`;
        tip.hidden = false; const wr = wrap.getBoundingClientRect();
        tip.style.left = Math.min(ev.clientX - wr.left + 10, wr.width - tip.offsetWidth - 4) + "px"; tip.style.top = Math.max(4, ev.clientY - wr.top - tip.offsetHeight - 8) + "px";
      });
      c.addEventListener("pointerleave", () => { tip.hidden = true; });
    });
  };

  /* trayectorias x–y con contornos del potencial (opcional) */
  window.trajChart = function (host, o) {
    if (typeof host === "string") host = document.getElementById(host);
    const W = o.w || 300, H = o.h || 300, pad = 8;
    host.classList.add("chart");
    if (o.title) { const h = document.createElement("div"); h.className = "chart-title small"; h.textContent = o.title; host.appendChild(h); }
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title || "trayectorias" }, null); host.appendChild(svg);
    const [x0, x1, y0, y1] = o.box;
    const s = Math.min((W - 2 * pad) / (x1 - x0), (H - 2 * pad) / (y1 - y0));
    const X = (x) => W / 2 + (x - (x0 + x1) / 2) * s, Y = (y) => H / 2 - (y - (y0 + y1) / 2) * s;
    el("rect", { x: X(x0), y: Y(y1), width: (x1 - x0) * s, height: (y1 - y0) * s, class: "panel" }, svg);
    const clipId = "t" + Math.random().toString(36).slice(2);
    el("rect", { x: X(x0), y: Y(y1), width: (x1 - x0) * s, height: (y1 - y0) * s }, el("clipPath", { id: clipId }, el("defs", {}, svg)));
    const g = el("g", { "clip-path": `url(#${clipId})` }, svg);
    for (const c of o.contours || []) el("path", { d: c.map((seg) => seg.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join("")).join(""), class: "contour" }, g);
    const path = (pts) => pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join("");
    (o.paths || []).forEach((p, k) => {
      const col = o.colors[k % o.colors.length];
      el("path", { d: path(p.rk), fill: "none", class: "truth" }, g);
      el("path", { d: path(p.nn), fill: "none", stroke: col, "stroke-width": 1.8, "stroke-linejoin": "round", "stroke-linecap": "round" }, g);
      el("circle", { cx: X(p.rk[0][0]), cy: Y(p.rk[0][1]), r: 3.2, fill: col, stroke: "var(--bg)", "stroke-width": 1.2 }, g);
    });
  };
})();
