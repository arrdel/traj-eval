(function () {
  "use strict";

  const state = {
    summary: null,
    suiteCache: {},
    currentSuiteId: "dream_arcc_smoke64_reasoning",
    currentTrajectory: null,
    currentStep: 1,
    playTimer: null,
  };

  function fmtNum(val, digits = 4) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) {
      return "–";
    }
    return Number(val).toFixed(digits);
  }

  function fmtSigned(val, digits = 4) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) {
      return "–";
    }
    const n = Number(val);
    return (n >= 0 ? "+" : "") + n.toFixed(digits);
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function formatCanvasHtml(packedText) {
    if (!packedText) {
      return '<span class="mask-chip">[empty canvas]</span>';
    }
    const escaped = escapeHtml(packedText);
    return escaped
      .replace(/&lt;&lt;M:(\d+)&gt;&gt;/g, function (_, count) {
        const n = Number(count);
        return n === 1
          ? '<span class="mask-chip">[MASK]</span>'
          : '<span class="mask-chip">[MASK &times;' + n + "]</span>";
      })
      .replace(/&lt;\|(?:mdm_)?mask\|&gt;/g, '<span class="mask-chip">[MASK]</span>');
  }

  function getStepState(traj, stepNum) {
    const keyframes = traj.keyframes || [];
    if (!keyframes.length) {
      return {
        step: stepNum,
        prediction: null,
        reasoning_text: "",
        confidence: null,
        answer_committed: null,
        committed_reasoning_fraction: null,
      };
    }
    let chosen = keyframes[0];
    for (let i = 0; i < keyframes.length; i++) {
      if (keyframes[i].step <= stepNum) {
        chosen = keyframes[i];
      } else {
        break;
      }
    }
    return chosen;
  }

  async function loadSuiteTrajectories(suiteId) {
    if (state.suiteCache[suiteId]) {
      return state.suiteCache[suiteId];
    }
    const resp = await fetch("data/suites/" + encodeURIComponent(suiteId) + ".json");
    if (!resp.ok) {
      throw new Error("Failed to load suite " + suiteId);
    }
    const data = await resp.json();
    state.suiteCache[suiteId] = data.trajectories || [];
    return state.suiteCache[suiteId];
  }

  function stopPlayback() {
    if (state.playTimer) {
      clearInterval(state.playTimer);
      state.playTimer = null;
    }
    const btn = document.getElementById("btn-play");
    if (btn) {
      btn.innerHTML = "&#9654; Play";
    }
  }

  function togglePlayback() {
    if (state.playTimer) {
      stopPlayback();
      return;
    }
    const traj = state.currentTrajectory;
    if (!traj) return;
    const speedMs = Number(document.getElementById("speed-select").value) || 55;
    if (state.currentStep >= traj.n_steps) {
      state.currentStep = 1;
    }
    document.getElementById("btn-play").innerHTML = "&#10074;&#10074; Pause";
    state.playTimer = setInterval(function () {
      if (state.currentStep < traj.n_steps) {
        setStep(state.currentStep + 1);
      } else {
        stopPlayback();
      }
    }, speedMs);
  }

  function buildVerdictBarcode(traj) {
    const container = document.getElementById("verdict-barcode");
    container.innerHTML = "";
    const nSteps = traj.n_steps || 256;
    const gold = String(traj.gold_answer || "").trim();
    for (let s = 1; s <= nSteps; s++) {
      const st = getStepState(traj, s);
      const cell = document.createElement("div");
      cell.className = "barcode-cell ";
      if (st.prediction === null || st.prediction === undefined) {
        cell.className += "bc-null";
      } else if (String(st.prediction).trim() === gold) {
        cell.className += "bc-correct";
      } else {
        cell.className += "bc-wrong";
      }
      cell.dataset.step = String(s);
      cell.title = "Step " + s + ": verdict=" + (st.prediction ?? "null");
      cell.addEventListener("click", function () {
        stopPlayback();
        setStep(s);
      });
      container.appendChild(cell);
    }
  }

  function updateSliderMarkers(traj) {
    const nSteps = Math.max(1, traj.n_steps || 256);
    const m = traj.metrics || {};
    const tStab = m.convergence_step;
    const tCommit = m.answer_commitment_step;

    const stabEl = document.getElementById("slider-marker-stab");
    const commitEl = document.getElementById("slider-marker-commit");
    document.getElementById("marker-tstab-val").textContent = tStab ?? "None";
    document.getElementById("marker-tcommit-val").textContent = tCommit ?? "None";

    if (tStab && tStab >= 1 && tStab <= nSteps) {
      const pct = ((tStab - 1) / Math.max(1, nSteps - 1)) * 100;
      stabEl.style.left = "calc(" + pct.toFixed(2) + "% - 1px)";
      stabEl.style.display = "block";
    } else {
      stabEl.style.display = "none";
    }

    if (tCommit && tCommit >= 1 && tCommit <= nSteps) {
      const pct = ((tCommit - 1) / Math.max(1, nSteps - 1)) * 100;
      commitEl.style.left = "calc(" + pct.toFixed(2) + "% - 1px)";
      commitEl.style.display = "block";
    } else {
      commitEl.style.display = "none";
    }
  }

  function renderTrajectoryScorecard(traj) {
    const m = traj.metrics || {};
    const finalPred = traj.final_prediction ?? m.final_prediction ?? "None";
    const gold = traj.gold_answer ?? m.gold_answer ?? "–";
    const correct = m.correct !== undefined ? m.correct : String(finalPred) === String(gold);

    document.getElementById("m-final-gold").textContent = finalPred + " / " + gold;
    const tag = document.getElementById("m-correct-tag");
    tag.textContent = correct ? "✓ Matches Gold Answer" : "✗ Does Not Match Gold";
    tag.style.color = correct ? "#10b981" : "#f59e0b";

    document.getElementById("m-tstab").textContent =
      m.convergence_step !== null && m.convergence_step !== undefined
        ? "Step " + m.convergence_step
        : "Undecided";
    document.getElementById("m-tstab-frac").textContent =
      m.convergence_fraction !== null && m.convergence_fraction !== undefined
        ? (m.convergence_fraction * 100).toFixed(1) + "% of horizon (" + (traj.n_steps || 256) + ")"
        : "–";

    document.getElementById("m-tcommit").textContent =
      m.answer_commitment_step !== null && m.answer_commitment_step !== undefined
        ? "Step " + m.answer_commitment_step
        : "None";
    document.getElementById("m-lag").textContent =
      m.stability_commitment_lag !== null && m.stability_commitment_lag !== undefined
        ? "Lag (T_commit − T_stab): " + m.stability_commitment_lag
        : "Lag: –";

    document.getElementById("m-unwritten").textContent = fmtNum(
      m.unwritten_reasoning_at_convergence,
      4
    );
    document.getElementById("m-order-gap").textContent = fmtSigned(m.order_gap, 4);
    document.getElementById("m-flip-rate").textContent = fmtNum(m.flip_rate, 4);
    document.getElementById("m-consistency").textContent =
      "Step Consistency: " + fmtNum(m.consistency, 4);

    const ef = Number(m.early_factor || 0);
    const pcr = Number(m.post_change_rate || 0);
    const lg = Number(m.length_growth || 0);

    document.getElementById("m-early-factor").textContent = fmtNum(ef, 4);
    document.getElementById("bar-early-factor").style.width = Math.min(100, ef * 100) + "%";

    document.getElementById("m-post-change").textContent = fmtNum(pcr, 4);
    document.getElementById("bar-post-change").style.width = Math.min(100, pcr * 100) + "%";

    document.getElementById("m-length-growth").textContent = fmtNum(lg, 4);
    document.getElementById("bar-length-growth").style.width = Math.min(100, lg * 100) + "%";

    document.getElementById("m-rat-score").textContent = fmtNum(m.rationalization_score, 4);
    document.getElementById("m-gated-score").textContent = fmtNum(
      m.gated_rationalization_score,
      4
    );

    const badge = document.getElementById("regime-classification-badge");
    const gated = Number(m.gated_rationalization_score || 0);
    const unwritten = Number(m.unwritten_reasoning_at_convergence || 0);
    if (gated >= 0.25 || unwritten >= 0.35) {
      badge.textContent = "Post-Convergence Backfill / Rationalization";
      badge.className = "badge badge-warn";
    } else if (ef >= 0.85 && pcr < 0.05) {
      badge.textContent = "Immediate Direct Commit";
      badge.className = "badge badge-neutral";
    } else {
      badge.textContent = "Rationale-First Deliberation";
      badge.className = "badge badge-success";
    }
  }

  function setStep(stepNum) {
    const traj = state.currentTrajectory;
    if (!traj) return;
    const nSteps = traj.n_steps || 256;
    const clamped = Math.max(1, Math.min(nSteps, Number(stepNum) || 1));
    state.currentStep = clamped;

    const slider = document.getElementById("step-slider");
    slider.max = String(nSteps);
    slider.value = String(clamped);

    document.getElementById("step-current").textContent = String(clamped);
    document.getElementById("step-total").textContent = String(nSteps);
    document.getElementById("canvas-step-num").textContent = String(clamped);

    const st = getStepState(traj, clamped);
    const m = traj.metrics || {};
    const tStab = m.convergence_step;

    const phaseBadge = document.getElementById("step-phase-badge");
    if (tStab && clamped >= tStab) {
      phaseBadge.textContent = "Verdict Locked In (t ≥ T_stab=" + tStab + ")";
      phaseBadge.className = "badge badge-success";
    } else {
      phaseBadge.textContent = "Pre-Convergence Exploration";
      phaseBadge.className = "badge badge-neutral";
    }

    document.getElementById("live-verdict").textContent =
      st.prediction !== null && st.prediction !== undefined ? String(st.prediction) : "null (Masked)";
    document.getElementById("live-gold").textContent = String(traj.gold_answer ?? "–");
    document.getElementById("live-committed").textContent =
      st.answer_committed === true
        ? "Yes (Unmasked)"
        : st.answer_committed === false
        ? "No (Masked)"
        : "–";

    const fr =
      st.committed_reasoning_fraction !== null && st.committed_reasoning_fraction !== undefined
        ? Number(st.committed_reasoning_fraction)
        : 0;
    document.getElementById("live-fr-val").textContent = (fr * 100).toFixed(1) + "%";
    document.getElementById("live-fr-bar").style.width = Math.min(100, fr * 100) + "%";

    document.getElementById("step-confidence-readout").textContent =
      st.confidence !== null && st.confidence !== undefined
        ? "Step Confidence: " + Number(st.confidence).toFixed(4)
        : "";

    document.getElementById("denoising-canvas").innerHTML = formatCanvasHtml(st.reasoning_text);

    const cells = document.querySelectorAll("#verdict-barcode .barcode-cell");
    cells.forEach(function (c) {
      if (Number(c.dataset.step) === clamped) {
        c.classList.add("bc-active");
      } else {
        c.classList.remove("bc-active");
      }
    });
  }

  async function selectSuiteAndItem(suiteId, preferredItemId) {
    stopPlayback();
    state.currentSuiteId = suiteId;
    const suiteSelect = document.getElementById("suite-select");
    if (suiteSelect.value !== suiteId) {
      suiteSelect.value = suiteId;
    }

    const trajs = await loadSuiteTrajectories(suiteId);
    const itemSelect = document.getElementById("item-select");
    itemSelect.innerHTML = "";
    trajs.forEach(function (t) {
      const opt = document.createElement("option");
      opt.value = t.id;
      const m = t.metrics || {};
      const statusSym = m.correct ? "✓" : "✗";
      opt.textContent =
        t.id +
        " (" +
        statusSym +
        " pred=" +
        (t.final_prediction ?? "null") +
        ", T_stab=" +
        (m.convergence_step ?? "–") +
        ", gated=" +
        fmtNum(m.gated_rationalization_score, 3) +
        ")";
      itemSelect.appendChild(opt);
    });

    let chosen = trajs[0];
    if (preferredItemId) {
      const match = trajs.find(function (t) {
        return t.id === preferredItemId;
      });
      if (match) chosen = match;
    }
    if (chosen) {
      itemSelect.value = chosen.id;
      loadTrajectoryObject(chosen);
    }
  }

  function loadTrajectoryObject(traj) {
    state.currentTrajectory = traj;
    document.getElementById("trajectory-prompt").textContent =
      traj.prompt || "(Custom JSONL trajectory — prompt not embedded in JSONL row)";
    buildVerdictBarcode(traj);
    updateSliderMarkers(traj);
    renderTrajectoryScorecard(traj);
    const initStep =
      (traj.metrics && traj.metrics.convergence_step) || Math.min(16, traj.n_steps || 1);
    setStep(initStep);
  }

  function renderLeaderboard() {
    const scaleFilter = document.getElementById("filter-scale").value;
    const taskFilter = document.getElementById("filter-task").value;
    const tbody = document.getElementById("leaderboard-tbody");
    tbody.innerHTML = "";

    const suites = (state.summary && state.summary.suites) || [];
    suites.forEach(function (s) {
      if (scaleFilter !== "all" && String(s.scale) !== scaleFilter) return;
      if (taskFilter !== "all" && s.task !== taskFilter) return;

      const agg = s.aggregate || {};
      const tr = document.createElement("tr");
      tr.innerHTML =
        "<td><strong>" +
        escapeHtml(s.model_label) +
        "</strong><br><small style='color:#9ca3af'><code>" +
        escapeHtml(s.hf_model_id) +
        "</code></small></td>" +
        "<td>" +
        escapeHtml(s.task_label) +
        "<br><small style='color:#9ca3af'>" +
        escapeHtml(s.condition_label) +
        "</small></td>" +
        "<td class='num'>" +
        s.scale +
        "</td>" +
        "<td class='num'><strong>" +
        (Number(agg.accuracy || 0) * 100).toFixed(2) +
        "%</strong></td>" +
        "<td class='num'>" +
        (Number(agg.coverage || 0) * 100).toFixed(1) +
        "%</td>" +
        "<td class='num'>" +
        fmtNum(agg.mean_flip_rate, 4) +
        "</td>" +
        "<td class='num'>" +
        fmtNum(agg.mean_convergence_step, 1) +
        "</td>" +
        "<td class='num'>" +
        fmtNum(agg.mean_answer_commitment_step, 1) +
        "</td>" +
        "<td class='num'>" +
        fmtNum(agg.mean_unwritten_reasoning_at_convergence, 4) +
        "</td>" +
        "<td class='num'>" +
        fmtSigned(agg.mean_order_gap, 4) +
        "</td>" +
        "<td class='num'><strong>" +
        fmtNum(agg.mean_gated_rationalization_score, 4) +
        "</strong></td>" +
        "<td><button class='btn btn-sm btn-outline inspect-suite-btn' data-suite='" +
        escapeHtml(s.suite_id) +
        "'>Scrub &rarr;</button></td>";
      tbody.appendChild(tr);
    });

    tbody.querySelectorAll(".inspect-suite-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        const sid = btn.getAttribute("data-suite");
        selectSuiteAndItem(sid, null);
        document.getElementById("explorer").scrollIntoView({ behavior: "smooth" });
      });
    });
  }

  function renderComparisons() {
    const grid = document.getElementById("comparisons-grid");
    grid.innerHTML = "";
    const comps = (state.summary && state.summary.comparisons) || [];
    comps.forEach(function (c) {
      const d = c.data || {};
      const accDelta = (d.accuracy_delta && d.accuracy_delta.delta) || 0;
      const baseAcc = (d.accuracy_delta && d.accuracy_delta.base_accuracy) || 0;
      const targAcc = (d.accuracy_delta && d.accuracy_delta.target_accuracy) || 0;
      const baseAgg = d.base_aggregate || {};
      const targAgg = d.target_aggregate || {};
      const deltaClass = accDelta >= 0 ? "delta-pos" : "delta-neg";

      const card = document.createElement("div");
      card.className = "comp-card";
      card.innerHTML =
        "<h4>" +
        escapeHtml(c.model_label) +
        "</h4>" +
        "<div class='comp-sub'>" +
        escapeHtml(c.task_label) +
        " &middot; <code>" +
        escapeHtml(d.base_label || "base") +
        "</code> &rarr; <code>" +
        escapeHtml(d.target_label || "target") +
        "</code></div>" +
        "<div class='comp-row'><span>Accuracy</span><span>" +
        (baseAcc * 100).toFixed(2) +
        "% &rarr; " +
        (targAcc * 100).toFixed(2) +
        "% (<span class='" +
        deltaClass +
        "'>" +
        (accDelta >= 0 ? "+" : "") +
        (accDelta * 100).toFixed(2) +
        " pp</span>)</span></div>" +
        "<div class='comp-row'><span>Mean <em>T</em><sub>stab</sub></span><span>" +
        fmtNum(baseAgg.mean_convergence_step, 1) +
        " &rarr; " +
        fmtNum(targAgg.mean_convergence_step, 1) +
        "</span></div>" +
        "<div class='comp-row'><span>Mean <em>T</em><sub>commit</sub></span><span>" +
        fmtNum(baseAgg.mean_answer_commitment_step, 1) +
        " &rarr; " +
        fmtNum(targAgg.mean_answer_commitment_step, 1) +
        "</span></div>" +
        "<div class='comp-row'><span>Order Gap</span><span>" +
        fmtSigned(baseAgg.mean_order_gap, 4) +
        " &rarr; " +
        fmtSigned(targAgg.mean_order_gap, 4) +
        "</span></div>" +
        "<div class='comp-row'><span>Gated Rationalization</span><span>" +
        fmtNum(baseAgg.mean_gated_rationalization_score, 4) +
        " &rarr; <strong>" +
        fmtNum(targAgg.mean_gated_rationalization_score, 4) +
        "</strong></span></div>";
      grid.appendChild(card);
    });
  }

  function renderScatter() {
    const suiteId = document.getElementById("scatter-suite").value;
    const xKey = document.getElementById("scatter-x").value;
    const yKey = document.getElementById("scatter-y").value;
    const svg = document.getElementById("scatter-svg");
    svg.innerHTML = "";

    const suites = (state.summary && state.summary.suites) || [];
    const suite = suites.find(function (s) {
      return s.suite_id === suiteId;
    });
    if (!suite || !suite.per_trajectory) return;

    const rows = suite.per_trajectory.filter(function (r) {
      return r[xKey] !== null && r[xKey] !== undefined && r[yKey] !== null && r[yKey] !== undefined;
    });
    if (!rows.length) return;

    const padL = 62,
      padR = 28,
      padT = 24,
      padB = 42;
    const W = 760,
      H = 340;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const xs = rows.map(function (r) {
      return Number(r[xKey]);
    });
    const ys = rows.map(function (r) {
      return Number(r[yKey]);
    });
    let xMin = Math.min.apply(null, xs),
      xMax = Math.max.apply(null, xs);
    let yMin = Math.min.apply(null, ys),
      yMax = Math.max.apply(null, ys);
    if (xMax === xMin) xMax = xMin + 1;
    if (yMax === yMin) yMax = yMin + 1;

    const ns = "http://www.w3.org/2000/svg";
    for (let i = 0; i <= 4; i++) {
      const gy = padT + (plotH * i) / 4;
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", String(padL));
      line.setAttribute("x2", String(W - padR));
      line.setAttribute("y1", String(gy));
      line.setAttribute("y2", String(gy));
      line.setAttribute("stroke", "#1f2b44");
      svg.appendChild(line);

      const yVal = yMax - ((yMax - yMin) * i) / 4;
      const lbl = document.createElementNS(ns, "text");
      lbl.setAttribute("x", String(padL - 8));
      lbl.setAttribute("y", String(gy + 4));
      lbl.setAttribute("text-anchor", "end");
      lbl.setAttribute("fill", "#9ca3af");
      lbl.setAttribute("font-size", "11");
      lbl.textContent = yVal.toFixed(2);
      svg.appendChild(lbl);
    }

    for (let i = 0; i <= 4; i++) {
      const gx = padL + (plotW * i) / 4;
      const xVal = xMin + ((xMax - xMin) * i) / 4;
      const lbl = document.createElementNS(ns, "text");
      lbl.setAttribute("x", String(gx));
      lbl.setAttribute("y", String(H - 12));
      lbl.setAttribute("text-anchor", "middle");
      lbl.setAttribute("fill", "#9ca3af");
      lbl.setAttribute("font-size", "11");
      lbl.textContent = xVal.toFixed(2);
      svg.appendChild(lbl);
    }

    rows.forEach(function (r) {
      const cx = padL + ((Number(r[xKey]) - xMin) / (xMax - xMin)) * plotW;
      const cy = padT + plotH - ((Number(r[yKey]) - yMin) / (yMax - yMin)) * plotH;
      const circle = document.createElementNS(ns, "circle");
      circle.setAttribute("cx", cx.toFixed(1));
      circle.setAttribute("cy", cy.toFixed(1));
      circle.setAttribute("r", "5.5");
      circle.setAttribute("fill", r.correct ? "#10b981" : "#f59e0b");
      circle.setAttribute("fill-opacity", "0.85");
      circle.setAttribute("stroke", "#0b0f19");
      circle.setAttribute("stroke-width", "1.2");
      circle.style.cursor = "pointer";

      const title = document.createElementNS(ns, "title");
      title.textContent =
        r.id +
        " | pred=" +
        r.final_prediction +
        " (gold=" +
        r.gold_answer +
        ") | " +
        xKey +
        "=" +
        r[xKey] +
        ", " +
        yKey +
        "=" +
        r[yKey];
      circle.appendChild(title);

      circle.addEventListener("click", function () {
        selectSuiteAndItem(suiteId, r.id);
        document.getElementById("explorer").scrollIntoView({ behavior: "smooth" });
      });
      svg.appendChild(circle);
    });
  }

  function scoreCustomTrajectoryClientSide(row) {
    const steps = row.steps || [];
    const nSteps = steps.length;
    const finalPred =
      row.final_prediction ?? (nSteps ? steps[nSteps - 1].prediction : null);
    const gold = row.gold_answer ?? null;
    const correct = finalPred !== null && gold !== null && String(finalPred) === String(gold);

    let tStab = null;
    if (finalPred !== null && finalPred !== undefined) {
      tStab = nSteps;
      for (let i = nSteps - 1; i >= 0; i--) {
        if (steps[i].prediction === finalPred) {
          tStab = steps[i].step;
        } else {
          break;
        }
      }
    }

    let tCommit = null;
    for (let i = 0; i < nSteps; i++) {
      if (steps[i].answer_committed === true) {
        tCommit = steps[i].step;
        break;
      }
    }

    const nonNull = steps
      .map(function (s) {
        return s.prediction;
      })
      .filter(function (p) {
        return p !== null && p !== undefined;
      });
    let flips = 0;
    for (let i = 1; i < nonNull.length; i++) {
      if (nonNull[i] !== nonNull[i - 1]) flips++;
    }
    const flipRate = nonNull.length > 1 ? flips / (nonNull.length - 1) : 0;

    return {
      id: String(row.id || "custom_item"),
      prompt: String(row.prompt || row.question || ""),
      gold_answer: gold,
      final_prediction: finalPred,
      n_steps: nSteps,
      metrics: {
        id: String(row.id || "custom_item"),
        final_prediction: finalPred,
        gold_answer: gold,
        correct: correct,
        convergence_step: tStab,
        convergence_fraction: tStab && nSteps ? tStab / nSteps : null,
        answer_commitment_step: tCommit,
        stability_commitment_lag: tStab && tCommit ? tCommit - tStab : null,
        flip_rate: flipRate,
        consistency: 1 - flipRate,
        unwritten_reasoning_at_convergence:
          tStab && steps[tStab - 1] && steps[tStab - 1].committed_reasoning_fraction !== null
            ? 1 - Number(steps[tStab - 1].committed_reasoning_fraction)
            : null,
        order_gap: null,
        early_factor: tStab && nSteps ? 1 - tStab / nSteps : 0,
        post_change_rate: 0,
        length_growth: 0,
        rationalization_score: 0,
        gated_rationalization_score: 0,
      },
      keyframes: steps.map(function (s) {
        return {
          step: Number(s.step),
          prediction: s.prediction,
          reasoning_text: s.reasoning_text || "",
          confidence: s.confidence,
          answer_committed: s.answer_committed,
          committed_reasoning_fraction: s.committed_reasoning_fraction,
        };
      }),
    };
  }

  function setupCustomUpload() {
    const fileInput = document.getElementById("jsonl-file-input");
    const statusEl = document.getElementById("upload-status");
    fileInput.addEventListener("change", function (ev) {
      const file = ev.target.files && ev.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function (e) {
        try {
          const text = String(e.target.result || "").trim();
          const lines = text.split(/\r?\n/).filter(Boolean);
          const customTrajs = lines.map(function (line) {
            return scoreCustomTrajectoryClientSide(JSON.parse(line));
          });
          if (!customTrajs.length) {
            statusEl.textContent = "File contained 0 trajectories.";
            return;
          }
          const customSuiteId = "custom_uploaded_jsonl";
          state.suiteCache[customSuiteId] = customTrajs;
          const suiteSelect = document.getElementById("suite-select");
          let opt = suiteSelect.querySelector("option[value='" + customSuiteId + "']");
          if (!opt) {
            opt = document.createElement("option");
            opt.value = customSuiteId;
            suiteSelect.prepend(opt);
          }
          opt.textContent = "Custom Uploaded File: " + file.name + " (" + customTrajs.length + " items)";
          statusEl.textContent =
            "Loaded " + customTrajs.length + " trajectories from " + file.name + ". Opened in Scrubber above!";
          selectSuiteAndItem(customSuiteId, customTrajs[0].id);
          document.getElementById("explorer").scrollIntoView({ behavior: "smooth" });
        } catch (err) {
          statusEl.textContent = "Error parsing JSONL: " + err.message;
        }
      };
      reader.readAsText(file);
    });
  }

  async function init() {
    const resp = await fetch("data/summary.json");
    state.summary = await resp.json();

    const suiteSelect = document.getElementById("suite-select");
    const scatterSuiteSelect = document.getElementById("scatter-suite");
    suiteSelect.innerHTML = "";
    scatterSuiteSelect.innerHTML = "";

    (state.summary.suites || []).forEach(function (s) {
      const label =
        s.model_label +
        " · " +
        s.task_label +
        " · " +
        s.condition_label +
        " (N=" +
        s.scale +
        ", acc=" +
        (Number(s.aggregate.accuracy || 0) * 100).toFixed(1) +
        "%)";
      const opt1 = document.createElement("option");
      opt1.value = s.suite_id;
      opt1.textContent = label;
      suiteSelect.appendChild(opt1);

      const opt2 = document.createElement("option");
      opt2.value = s.suite_id;
      opt2.textContent = label;
      scatterSuiteSelect.appendChild(opt2);
    });

    scatterSuiteSelect.value = "llada_15_gsm8k_smoke64";

    suiteSelect.addEventListener("change", function () {
      selectSuiteAndItem(suiteSelect.value, null);
    });

    document.getElementById("item-select").addEventListener("change", function (ev) {
      stopPlayback();
      const trajs = state.suiteCache[state.currentSuiteId] || [];
      const found = trajs.find(function (t) {
        return t.id === ev.target.value;
      });
      if (found) loadTrajectoryObject(found);
    });

    document.getElementById("step-slider").addEventListener("input", function (ev) {
      stopPlayback();
      setStep(Number(ev.target.value));
    });

    document.getElementById("btn-play").addEventListener("click", togglePlayback);
    document.getElementById("btn-prev").addEventListener("click", function () {
      stopPlayback();
      setStep(state.currentStep - 1);
    });
    document.getElementById("btn-next").addEventListener("click", function () {
      stopPlayback();
      setStep(state.currentStep + 1);
    });
    document.getElementById("btn-jump-stab").addEventListener("click", function () {
      stopPlayback();
      const m = (state.currentTrajectory && state.currentTrajectory.metrics) || {};
      if (m.convergence_step) setStep(m.convergence_step);
    });
    document.getElementById("btn-jump-commit").addEventListener("click", function () {
      stopPlayback();
      const m = (state.currentTrajectory && state.currentTrajectory.metrics) || {};
      if (m.answer_commitment_step) setStep(m.answer_commitment_step);
    });

    document.querySelectorAll(".chip-btn[data-preset-suite]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        document.querySelectorAll(".chip-btn").forEach(function (b) {
          b.classList.remove("active");
        });
        btn.classList.add("active");
        selectSuiteAndItem(
          btn.getAttribute("data-preset-suite"),
          btn.getAttribute("data-preset-item")
        );
      });
    });

    document.getElementById("filter-scale").addEventListener("change", renderLeaderboard);
    document.getElementById("filter-task").addEventListener("change", renderLeaderboard);
    document.getElementById("scatter-suite").addEventListener("change", renderScatter);
    document.getElementById("scatter-x").addEventListener("change", renderScatter);
    document.getElementById("scatter-y").addEventListener("change", renderScatter);

    renderLeaderboard();
    renderComparisons();
    renderScatter();
    setupCustomUpload();

    await selectSuiteAndItem("dream_arcc_smoke64_reasoning", "Mercury_407327");
  }

  init();
})();
