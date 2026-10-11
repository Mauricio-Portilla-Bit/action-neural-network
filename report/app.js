/* Monta las figuras del reporte a partir de DATA (inyectado por make_report.py). */
(function () {
  const C = { s1: "var(--s1)", s2: "var(--s2)", s3: "var(--s3)", s4: "var(--s4)", s5: "var(--s5)", s6: "var(--s6)" };
  const $ = (id) => document.getElementById(id);
  const clear = (id) => { const e = $(id); if (e) { e.innerHTML = ""; e.className = ""; } return e; };
  const D = DATA;

  // 1. condicionamiento
  if (D.cond) lineChart("f_cond", {
    title: "Número de condición de la pérdida contra N", xlog: true, ylog: true, xlabel: "N (intervalos de tiempo)", xname: "N",
    xfmt: (v) => String(v), yr: [0.5, 1e11],
    series: [
      { name: "‖r‖²  residuo crudo (original)", color: C.s2, x: D.cond.N, y: D.cond.raw, markers: true },
      { name: "‖π‖²  balance de momento", color: C.s4, x: D.cond.N, y: D.cond.mom, markers: true },
      { name: "‖c‖²  Picard", color: C.s1, x: D.cond.N, y: D.cond.pos, markers: true },
    ],
  });

  // 2. BVP
  if (D.bvp) {
    const ks = Object.keys(D.bvp);
    lineChart("f_bvp", {
      title: "Acción media durante el entrenamiento (la pérdida es S)", xlabel: "paso de entrenamiento", xname: "paso", xfmt: (v) => String(v),
      yr: [-6, 1.5], yticks: 5,
      series: [
        { name: "ωT < π : mínimo", color: C.s1, x: D.bvp[ks[0]].step, y: D.bvp[ks[0]].S },
        { name: "ωT > π : punto de silla", color: C.s2, x: D.bvp[ks[1]].step, y: D.bvp[ks[1]].S },
      ],
    });
  }

  // 3. proyectil
  const P = D.proj || {};
  const pser = [["proj_orig", "Original", C.s2], ["proj_traj_raw", "Red nueva + pérdida cruda", C.s4],
    ["proj_orignet_pos", "Red original + pérdida de Picard", C.s3], ["proj_traj_pos", "Red nueva + pérdida de Picard", C.s1]]
    .filter((s) => P[s[0]]).map((s) => ({ name: s[1], color: s[2], x: P[s[0]].x, y: P[s[0]].y }));
  if (pser.length) lineChart("f_proj", {
    title: "Tiro parabólico: RMSE (m) contra pasos de entrenamiento", xlog: true, ylog: true, xlabel: "paso de entrenamiento", xname: "paso",
    xfmt: (v) => String(Math.round(v)), ylabel: "RMSE (m)", series: pser, h: 360, w: 760,
  });

  // 4. ablación
  const A = D.abl || {};
  const an = { raw: ["Cruda ‖r‖²", C.s2], mom: ["Momento ‖π‖²", C.s4], posmom: ["Picard + momento (peso 1)", C.s3], posmom01: ["Picard + momento (peso 0.01)", C.s5], pos: ["Picard ‖c‖² (gradiente completo)", C.s1], sg1: ["Picard (semigradiente)", C.s6] };
  const aser = Object.keys(an).filter((k) => A[k]).map((k) => ({ name: an[k][0], color: an[k][1], x: A[k].x, y: A[k].y, markers: true }));
  if (aser.length) lineChart("f_abl", { title: "Péndulo: RMSE final (rad) contra N", xlog: true, ylog: true, xlabel: "N", xname: "N", xfmt: (v) => String(v), series: aser, xr: [22, 230] });
  const AC = D.ablc || {};
  const acs = ["raw", "mom", "pos"].filter((k) => AC[k]).map((k) => ({ name: an[k][0], color: an[k][1], x: AC[k].x, y: AC[k].y }));
  if (acs.length) lineChart("f_ablc", { title: "Péndulo, N = 100: RMSE (rad) contra pasos", xlog: true, ylog: true, xlabel: "paso de entrenamiento", xname: "paso", xfmt: (v) => String(Math.round(v)), series: acs });

  if (D.dp_cmp) lineChart("f_dp", { title: "Péndulo doble: gradiente completo contra semigradiente", ylog: true, xlabel: "paso de entrenamiento", xname: "paso", xfmt: (v) => String(Math.round(v)), w: 760, h: 300,
    series: [{ name: "Gradiente completo", color: C.s2, x: D.dp_cmp.pos.x, y: D.dp_cmp.pos.y }, { name: "Semigradiente", color: C.s1, x: D.dp_cmp.sg.x, y: D.dp_cmp.sg.y }] });

  // 5. sistemas
  const modes = [["nn", "Sólo red", C.s1], ["int", "Red + impulso", C.s3], ["1", "+1 Picard", C.s6], ["2", "+2 Picard", C.s5]];
  function tabs(id, items, onPick) {
    const host = $(id); if (!host) return;
    items.forEach(([key, label], i) => {
      const b = document.createElement("button"); b.type = "button"; b.textContent = label; b.id = id + "_" + key;
      b.setAttribute("role", "tab"); b.setAttribute("aria-selected", i === 0 ? "true" : "false");
      b.addEventListener("click", () => { host.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", "false")); b.setAttribute("aria-selected", "true"); onPick(key); });
      host.appendChild(b);
    });
    if (items.length) onPick(items[0][0]);
  }
  function rollSeries(o) {
    return modes.filter((m) => o["roll_" + m[0]]).map((m, i) => ({
      name: m[1], color: m[2], x: o.t, y: o["roll_" + m[0]][1], band: i === 0 ? [o["roll_" + m[0]][0], o["roll_" + m[0]][2]] : null,
    }));
  }
  function picSeries(o) {
    const it = o.picard_nn.map((_, i) => i);
    return [{ name: "desde la red", color: C.s1, x: it, y: o.picard_nn, markers: true }, { name: "desde vuelo libre", color: C.s2, x: it, y: o.picard_free, markers: true }];
  }
  const S = D.systems || {};
  function showSys(k) {
    const s = S[k];
    clear("f_roll"); clear("f_traj"); clear("f_dE"); clear("f_pic");
    if (s.t) {
      lineChart("f_roll", { title: s.label + ": error de posición contra tiempo", ylog: true, xlabel: "t", series: rollSeries(s) });
      lineChart("f_dE", { title: s.label + ": |E − E₀| contra tiempo", ylog: true, xlabel: "t",
        series: modes.filter((m) => s["dE_" + m[0]]).map((m) => ({ name: m[1], color: m[2], x: s.tw, y: s["dE_" + m[0]][1] })) });
      const d = s.ex_nn[0][0].length, tt = s.ex_nn[0].map((_, i) => (i + 1) * s.ex_dt);
      const twoD = k === "henon_heiles" || k === "magnetic";
      if (twoD) {
        const host = $("f_traj"); host.className = "trajgrid";
        const all = s.ex_rk.slice(0, 4).flat(), xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
        const r = Math.max(Math.max(...xs.map(Math.abs)), Math.max(...ys.map(Math.abs))) * 1.08;
        s.ex_nn.slice(0, 4).forEach((nn, i) => {
          const dv = document.createElement("div"); host.appendChild(dv);
          trajChart(dv, { box: [-r, r, -r, r], colors: [C.s1], paths: [{ nn, rk: s.ex_rk[i] }], w: 240, h: 240 });
        });
        $("cap_traj").textContent = "Cuatro trayectorias de prueba en el plano x–y hasta t = " + tt[tt.length - 1].toFixed(0) + " (modo «sólo red»). Color: red. Punteado: RK4.";
      } else {
        const ser = [];
        const cols = [C.s1, C.s3, C.s6];
        for (let i = 0; i < (d === 1 ? 3 : 1); i++) for (let j = 0; j < d; j++) {
          const col = d === 1 ? cols[i] : cols[j];
          ser.push({ name: d === 1 ? "red, ejemplo " + (i + 1) : "red, θ" + (j + 1), color: col, x: tt, y: s.ex_nn[i].map((p) => p[j]) });
          ser.push({ name: "RK4", color: "var(--ink)", dash: "2.5 3", width: 1.2, x: tt, y: s.ex_rk[i].map((p) => p[j]) });
        }
        lineChart("f_traj", { title: s.label + ": ángulo contra tiempo", xlabel: "t", series: ser });
        $("cap_traj").textContent = "Trayectorias de prueba (modo «sólo red») contra RK4 (punteado).";
      }
    }
    lineChart("f_pic", { title: s.label + ": RMSE tras k iteraciones de Picard", ylog: true, xlabel: "iteraciones de Picard", xname: "k", xfmt: (v) => String(v), series: picSeries(s) });
  }
  tabs("sys_tabs", Object.keys(S).map((k) => [k, S[k].label]), showSys);

  // 6. universal
  const U = (D.uni || {})[UNI_TAG];
  if (U) {
    const host = $("uni_traj"), cols = [C.s1, C.s2, C.s3, C.s6, C.s5];
    Object.keys(U.pots).forEach((k) => {
      const p = U.pots[k], dv = document.createElement("div"); host.appendChild(dv);
      trajChart(dv, { title: p.label, box: p.box, contours: (D.contours || {})[k], colors: cols, w: 260, h: 260,
        paths: p.ex_nn.map((nn, i) => ({ nn, rk: p.ex_rk[i] })) });
    });
    const showU = (k) => {
      const p = U.pots[k]; clear("f_uroll"); clear("f_upic");
      lineChart("f_uroll", { title: p.label + ": error de posición contra tiempo", ylog: true, xlabel: "t", series: rollSeries(p) });
      lineChart("f_upic", { title: p.label + ": RMSE tras k iteraciones de Picard", ylog: true, xlabel: "iteraciones de Picard", xname: "k", xfmt: (v) => String(v), series: picSeries(p) });
    };
    tabs("uni_tabs", Object.keys(U.pots).map((k) => [k, U.pots[k].label]), showU);
  }

  // 7. certificado
  const g1 = Object.keys(S).map((k, i) => ({ name: S[k].label, color: [C.s1, C.s2, C.s3, C.s6][i % 4], x: S[k].cert_x, y: S[k].cert_y }));
  if (g1.length) scatterChart("f_cert1", { title: "Sistemas con nombre", groups: g1, xlabel: "defecto de Picard  max‖c‖", ylabel: "error real máximo", xname: "defecto", yname: "error", legend: true, series: g1 });
  if (U) {
    const ks = Object.keys(U.pots);
    const g2 = ks.map((k, i) => ({ name: U.pots[k].label, color: [C.s1, C.s2, C.s3, C.s6, C.s5][i % 5], x: U.pots[k].cert_x, y: U.pots[k].cert_y }));
    scatterChart("f_cert2", { title: "Operador universal, potenciales nunca vistos", groups: g2, xlabel: "defecto de Picard  max‖c‖", ylabel: "error real máximo", xname: "defecto", yname: "error", series: g2 });
  }
})();
