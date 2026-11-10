(function () {
  "use strict";

  function loadEmbeddedData() {
    var el = document.getElementById("page-data");
    if (!el) {
      throw new Error("Missing #page-data JSON script element");
    }
    return JSON.parse(el.textContent);
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function renderTwoConditionsChart(comp) {
    var svg = document.getElementById("two-conditions-svg");
    if (!svg || !comp) return;
    svg.innerHTML = "";

    var steps = comp.sampled_steps;
    var rVerdicts = comp.reasoning.verdicts;
    var dVerdicts = comp.direct.verdicts;

    var W = 860;
    var H = 300;
    var padL = 95;
    var padR = 36;
    var padT = 32;
    var padB = 48;
    var plotW = W - padL - padR;
    var plotH = H - padT - padB;

    var yLevels = {
      null: padT + plotH * 0.82,
      A: padT + plotH * 0.48,
      C: padT + plotH * 0.14,
    };

    var ns = "http://www.w3.org/2000/svg";

    var yLabels = [
      { key: "C", label: "Verdict C (gold)" },
      { key: "A", label: "Verdict A" },
      { key: "null", label: "None (masked)" },
    ];

    yLabels.forEach(function (row) {
      var y = yLevels[row.key];
      var line = document.createElementNS(ns, "line");
      line.setAttribute("x1", String(padL));
      line.setAttribute("x2", String(W - padR));
      line.setAttribute("y1", String(y));
      line.setAttribute("y2", String(y));
      line.setAttribute("stroke", "#cbd5e1");
      line.setAttribute("stroke-dasharray", row.key === "C" ? "none" : "4,4");
      svg.appendChild(line);

      var txt = document.createElementNS(ns, "text");
      txt.setAttribute("x", String(padL - 10));
      txt.setAttribute("y", String(y + 4));
      txt.setAttribute("text-anchor", "end");
      txt.setAttribute("fill", row.key === "C" ? "#047857" : "#475569");
      txt.setAttribute("font-weight", row.key === "C" ? "600" : "400");
      txt.setAttribute("font-size", "11.5");
      txt.textContent = row.label;
      svg.appendChild(txt);
    });

    function xForStep(stepVal) {
      return padL + (stepVal / 255) * plotW;
    }

    steps.forEach(function (st) {
      var x = xForStep(st);
      var tick = document.createElementNS(ns, "line");
      tick.setAttribute("x1", String(x));
      tick.setAttribute("x2", String(x));
      tick.setAttribute("y1", String(padT + plotH));
      tick.setAttribute("y2", String(padT + plotH + 5));
      tick.setAttribute("stroke", "#94a3b8");
      svg.appendChild(tick);

      var lbl = document.createElementNS(ns, "text");
      lbl.setAttribute("x", String(x));
      lbl.setAttribute("y", String(H - 18));
      lbl.setAttribute("text-anchor", "middle");
      lbl.setAttribute("fill", "#475569");
      lbl.setAttribute("font-size", "11");
      lbl.textContent = String(st);
      svg.appendChild(lbl);
    });

    var axisTitle = document.createElementNS(ns, "text");
    axisTitle.setAttribute("x", String(padL + plotW / 2));
    axisTitle.setAttribute("y", String(H - 3));
    axisTitle.setAttribute("text-anchor", "middle");
    axisTitle.setAttribute("fill", "#475569");
    axisTitle.setAttribute("font-size", "11.5");
    axisTitle.textContent = "Denoising step (0 to 255, sampled every 16 steps)";
    svg.appendChild(axisTitle);

    function drawSeries(verdicts, color, yOffset, seriesLabel) {
      var pts = [];
      for (var i = 0; i < steps.length; i++) {
        var k = verdicts[i] === null ? "null" : String(verdicts[i]);
        var x = xForStep(steps[i]);
        var y = yLevels[k] + yOffset;
        pts.push(x.toFixed(1) + "," + y.toFixed(1));
      }
      var poly = document.createElementNS(ns, "polyline");
      poly.setAttribute("fill", "none");
      poly.setAttribute("stroke", color);
      poly.setAttribute("stroke-width", "2.6");
      poly.setAttribute("points", pts.join(" "));
      svg.appendChild(poly);

      for (var j = 0; j < steps.length; j++) {
        var key = verdicts[j] === null ? "null" : String(verdicts[j]);
        var cx = xForStep(steps[j]);
        var cy = yLevels[key] + yOffset;
        var c = document.createElementNS(ns, "circle");
        c.setAttribute("cx", cx.toFixed(1));
        c.setAttribute("cy", cy.toFixed(1));
        c.setAttribute("r", "4.8");
        c.setAttribute("fill", color);
        c.setAttribute("stroke", "#ffffff");
        c.setAttribute("stroke-width", "1.5");
        c.style.cursor = "pointer";
        (function (idx) {
          c.addEventListener("click", function () {
            var slider = document.getElementById("sample-step-slider");
            if (slider) {
              slider.value = String(idx);
              updateStepInspector(comp, idx);
            }
          });
        })(j);
        var title = document.createElementNS(ns, "title");
        title.textContent =
          seriesLabel +
          " | step " +
          steps[j] +
          ": verdict=" +
          (verdicts[j] === null ? "None" : verdicts[j]);
        c.appendChild(title);
        svg.appendChild(c);
      }
    }

    drawSeries(dVerdicts, "#0284c7", -4, "Direct (Mercury_7175875)");
    drawSeries(rVerdicts, "#ea580c", 4, "Reasoning (Mercury_LBS10610)");
  }

  function updateStepInspector(comp, sampleIndex) {
    var idx = Math.max(0, Math.min(comp.sampled_steps.length - 1, Number(sampleIndex) || 0));
    var stepNum = comp.sampled_steps[idx];
    document.getElementById("sample-step-label").textContent = String(stepNum);

    var rPred = comp.reasoning.verdicts[idx];
    var dPred = comp.direct.verdicts[idx];
    document.getElementById("reasoning-step-verdict").textContent =
      "Verdict: " + (rPred === null ? "None" : rPred);
    document.getElementById("direct-step-verdict").textContent =
      "Verdict: " + (dPred === null ? "None" : dPred);

    document.getElementById("reasoning-step-text").textContent = comp.reasoning.texts[idx];
    document.getElementById("direct-step-text").textContent = comp.direct.texts[idx];
  }

  function renderResultsTable(rows) {
    var tbody = document.getElementById("results-64-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    rows.forEach(function (r) {
      var tr = document.createElement("tr");
      tr.innerHTML =
        "<td><strong>" +
        escapeHtml(r.model) +
        "</strong></td>" +
        "<td>" +
        escapeHtml(r.task) +
        "</td>" +
        "<td><code>" +
        escapeHtml(r.condition) +
        "</code></td>" +
        "<td class='num'>" +
        r.n +
        "</td>" +
        "<td class='num'><strong>" +
        Number(r.accuracy).toFixed(4) +
        "</strong></td>" +
        "<td class='num'>" +
        Number(r.coverage).toFixed(4) +
        "</td>" +
        "<td class='num'>" +
        Number(r.conditional_accuracy).toFixed(4) +
        "</td>" +
        "<td class='num'>" +
        Number(r.mean_convergence_step).toFixed(1) +
        "</td>" +
        "<td>" +
        escapeHtml(r.note) +
        "</td>";
      tbody.appendChild(tr);
    });
  }

  function renderDreamRationalization(ratData) {
    var tbody = document.getElementById("dream-rat-tbody");
    var svg = document.getElementById("dream-rat-svg");
    if (!tbody || !svg || !ratData) return;

    tbody.innerHTML = "";
    ratData.components.forEach(function (c) {
      var tr = document.createElement("tr");
      tr.innerHTML =
        "<td><code>" +
        escapeHtml(c.label) +
        "</code></td>" +
        "<td class='num'>" +
        Number(c.direct).toFixed(4) +
        "</td>" +
        "<td class='num'>" +
        Number(c.reasoning).toFixed(4) +
        "</td>";
      tbody.appendChild(tr);
    });

    svg.innerHTML = "";
    var ns = "http://www.w3.org/2000/svg";
    var W = 460;
    var H = 250;
    var padL = 132;
    var padR = 64;
    var padT = 26;
    var rowH = 48;
    var barMaxW = W - padL - padR;

    ratData.components.forEach(function (c, idx) {
      var yBase = padT + idx * rowH;

      var lbl = document.createElementNS(ns, "text");
      lbl.setAttribute("x", String(padL - 10));
      lbl.setAttribute("y", String(yBase + 18));
      lbl.setAttribute("text-anchor", "end");
      lbl.setAttribute("fill", "#334155");
      lbl.setAttribute("font-weight", "600");
      lbl.setAttribute("font-size", "11.5");
      lbl.textContent = c.label;
      svg.appendChild(lbl);

      var wDirect = Math.max(2, Number(c.direct) * barMaxW);
      var rectD = document.createElementNS(ns, "rect");
      rectD.setAttribute("x", String(padL));
      rectD.setAttribute("y", String(yBase + 2));
      rectD.setAttribute("width", wDirect.toFixed(1));
      rectD.setAttribute("height", "12");
      rectD.setAttribute("rx", "3");
      rectD.setAttribute("fill", "#0284c7");
      svg.appendChild(rectD);

      var valD = document.createElementNS(ns, "text");
      valD.setAttribute("x", String(padL + wDirect + 6));
      valD.setAttribute("y", String(yBase + 12));
      valD.setAttribute("fill", "#0369a1");
      valD.setAttribute("font-size", "10.5");
      valD.textContent = "direct: " + Number(c.direct).toFixed(4);
      svg.appendChild(valD);

      var wReason = Math.max(2, Number(c.reasoning) * barMaxW);
      var rectR = document.createElementNS(ns, "rect");
      rectR.setAttribute("x", String(padL));
      rectR.setAttribute("y", String(yBase + 18));
      rectR.setAttribute("width", wReason.toFixed(1));
      rectR.setAttribute("height", "12");
      rectR.setAttribute("rx", "3");
      rectR.setAttribute("fill", "#ea580c");
      svg.appendChild(rectR);

      var valR = document.createElementNS(ns, "text");
      valR.setAttribute("x", String(padL + wReason + 6));
      valR.setAttribute("y", String(yBase + 28));
      valR.setAttribute("fill", "#c2410c");
      valR.setAttribute("font-size", "10.5");
      valR.textContent = "reasoning: " + Number(c.reasoning).toFixed(4);
      svg.appendChild(valR);
    });
  }

  function setupNavigation() {
    var navToggle = document.getElementById("nav-toggle");
    var links = document.querySelectorAll('a[href^="#"]');
    links.forEach(function (link) {
      link.addEventListener("click", function (ev) {
        var href = link.getAttribute("href");
        if (!href || href.length < 2) return;
        var targetId = href.slice(1);
        var targetEl = document.getElementById(targetId);
        if (!targetEl) return;
        ev.preventDefault();
        if (navToggle) {
          navToggle.checked = false;
        }
        targetEl.scrollIntoView({ behavior: "smooth", block: "start" });
        try {
          window.history.replaceState(null, "", href);
        } catch (_err) {
          // Ignore in sandboxed iframes (e.g. Hugging Face Static Space).
        }
      });
    });
  }

  function init() {
    setupNavigation();
    var data = loadEmbeddedData();
    renderTwoConditionsChart(data.trajectory_comparison);
    updateStepInspector(data.trajectory_comparison, 5);

    var slider = document.getElementById("sample-step-slider");
    if (slider) {
      slider.addEventListener("input", function (ev) {
        updateStepInspector(data.trajectory_comparison, Number(ev.target.value));
      });
    }

    renderResultsTable(data.results_64);
    renderDreamRationalization(data.dream_rationalization_side_by_side);
  }

  init();
})();
