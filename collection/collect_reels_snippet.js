// Reels collector: paste in the DevTools console on https://www.instagram.com/<creator>/reels/ while logged in.
// Step 1 records the JSON the page itself loads while you scroll (views, likes, comments).
// Step 2 ("Get details") asks Instagram, one reel at a time with 2 to 4 s pauses, for the caption,
// date and video link of the creator's own reels. It stops by itself on any rate-limit or login error.
// It never reads or sends credentials and saves the file only to your computer.
(() => {
  if (window.__reelsCollector) { console.log("Collector already running."); return; }
  const VIEW_KEYS = ["play_count", "ig_play_count", "view_count", "video_view_count", "video_play_count", "clips_play_count"];
  const items = new Map();
  const username = (location.pathname.split("/").filter(Boolean)[0] || "unknown").toLowerCase();
  let status = "scroll down, then click Get details";
  let running = false;

  const hasViews = (o) => VIEW_KEYS.some((k) => typeof o[k] === "number");
  const isOwn = (o) => (((o.user || o.owner || {}).username) || "").toLowerCase() === username;
  const score = (o) => Object.keys(o).length + (hasViews(o) ? 100 : 0) + (o.caption ? 10 : 0) + (o.video_versions ? 10 : 0);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function walk(node, depth = 0) {
    if (!node || typeof node !== "object" || depth > 40) return;
    if (Array.isArray(node)) { node.forEach((n) => walk(n, depth + 1)); return; }
    const code = node.code || node.shortcode;
    if (typeof code === "string" && (hasViews(node) || node.media_type === 2 || node.product_type === "clips" || items.has(code))) {
      const prev = items.get(code);
      if (!prev) items.set(code, node);
      else if (score(node) > score(prev)) items.set(code, Object.assign({}, prev, node));
      else items.set(code, Object.assign({}, node, prev));
    }
    for (const k in node) {
      const v = node[k];
      if (v && typeof v === "object") walk(v, depth + 1);
    }
  }

  function ingestText(text) {
    if (!text || typeof text !== "string") return;
    const clean = text.replace(/^for \(;;\);/, "").trim();
    if (!clean || (clean[0] !== "{" && clean[0] !== "[")) return;
    try { walk(JSON.parse(clean)); update(); return; } catch (e) { /* may be several JSON objects */ }
    clean.split("\n").forEach((line) => { try { walk(JSON.parse(line)); } catch (e) {} });
    update();
  }

  const origFetch = window.fetch;
  window.fetch = async function (...args) {
    const resp = await origFetch.apply(this, args);
    try { resp.clone().text().then(ingestText).catch(() => {}); } catch (e) {}
    return resp;
  };
  const origOpen = XMLHttpRequest.prototype.open;
  const origSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (...args) { this.__rcUrl = args[1]; return origOpen.apply(this, args); };
  XMLHttpRequest.prototype.send = function (...args) {
    this.addEventListener("load", function () {
      try {
        if (this.responseType === "" || this.responseType === "text") ingestText(this.responseText);
        else if (this.responseType === "json") { walk(this.response); update(); }
      } catch (e) {}
    });
    return origSend.apply(this, args);
  };

  const box = document.createElement("div");
  box.style.cssText = "position:fixed;right:16px;bottom:16px;z-index:999999;background:#0f6b6b;color:#fff;width:300px;" +
    "font:13px/1.4 system-ui,sans-serif;padding:10px 12px;border-radius:10px;box-shadow:0 4px 16px rgba(0,0,0,.25)";
  const label = document.createElement("div");
  const note = document.createElement("div");
  note.style.cssText = "opacity:.85;font-size:12px;margin-top:2px";
  const mkBtn = (t) => { const b = document.createElement("button"); b.textContent = t;
    b.style.cssText = "margin-top:6px;width:100%;border:0;border-radius:6px;padding:6px;cursor:pointer;font-weight:600"; return b; };
  const autoBtn = mkBtn("AUTO (scroll + details + save)");
  const detailsBtn = mkBtn("Get details");
  const saveBtn = mkBtn("Save JSON");
  const exportBtn = mkBtn("Export all (every profile saved in this browser)");
  box.append(label, note, autoBtn, detailsBtn, saveBtn, exportBtn);
  document.body.appendChild(box);

  function own() { return [...items.values()].filter(isOwn); }
  function update() {
    const o = own();
    label.textContent = `@${username}: ${o.length} own reels (${o.filter(hasViews).length} with views, ${o.filter((x) => x.__details).length} with details)`;
    note.textContent = status;
  }

  async function getDetails() {
    if (running) return;
    running = true;
    const todo = own().filter((o) => !o.__details && (o.pk || o.id));
    let fails = 0;
    status = `getting details: 0 of ${todo.length}`;
    update();
    for (let i = 0; i < todo.length; i++) {
      const o = todo[i];
      const pk = String(o.pk || o.id).split("_")[0];
      try {
        const r = await origFetch(`/api/v1/media/${pk}/info/`, {
          headers: { "x-ig-app-id": "936619743392459", "x-requested-with": "XMLHttpRequest" }, credentials: "include" });
        if ([401, 403, 429].includes(r.status)) { status = `STOPPED by Instagram (HTTP ${r.status}). Click Save JSON.`; break; }
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const j = await r.json();
        if (j.require_login || j.message === "checkpoint_required" || j.status === "fail") { status = "STOPPED: login or checkpoint required. Click Save JSON."; break; }
        const m = (j.items || [])[0];
        if (m && m.code) { walk(m); const cur = items.get(m.code); if (cur) cur.__details = true; fails = 0; }
        else throw new Error("empty");
      } catch (e) {
        fails++;
        if (fails >= 5) { status = "STOPPED after 5 errors in a row. Click Save JSON."; break; }
        status = `error on ${i + 1} of ${todo.length} (${e.message}), pausing ${30 * fails} s`;
        update();
        await sleep(30000 * fails);  // back off on transient errors instead of giving up
        continue;
      }
      status = `getting details: ${i + 1} of ${todo.length}`;
      update();
      await sleep(2000 + Math.random() * 2000);
    }
    if (status.startsWith("getting details")) status = "DETAILS DONE. Click Save JSON.";
    running = false;
    update();
  }

  async function restoreBackup() {
    // Reuse details already fetched in an earlier run of this profile, so a re-run only asks for the missing ones.
    try {
      const prev = (await idbAll()).find((r) => r.username === username);
      if (!prev) return 0;
      let n = 0;
      prev.items.forEach((it) => {
        if (!it.__details || !it.code) return;
        items.set(it.code, Object.assign({}, items.get(it.code) || {}, it));
        n++;
      });
      return n;
    } catch (e) { return 0; }
  }

  async function autoRun(target = 120) {
    if (running) return;
    const restored = await restoreBackup();
    if (restored) { status = `restored ${restored} reels with details from browser backup`; update(); }
    let last = -1, still = 0;
    while (own().length < target && still < 6) {
      status = `auto-scrolling: ${own().length} of ${target}`;
      update();
      window.scrollTo(0, document.documentElement.scrollHeight);
      await sleep(2500 + Math.random() * 1000);
      const n = own().length;
      still = n === last ? still + 1 : 0;
      last = n;
    }
    window.scrollTo(0, 0);
    await getDetails();
    try { await idbPut(username, JSON.stringify(makePayload())); } catch (e) { console.log("browser backup failed", e); }
    status = status.startsWith("STOPPED") ? status + " Backup kept in browser. Click Save JSON."
                                          : "ALL DONE. Backup kept in browser. Click Save JSON.";
    update();
  }

  function idbOpen() {
    return new Promise((res, rej) => {
      const req = indexedDB.open("reelsCollector", 1);
      req.onupgradeneeded = () => req.result.createObjectStore("runs");
      req.onsuccess = () => res(req.result);
      req.onerror = () => rej(req.error);
    });
  }
  async function idbPut(key, value) {
    const db = await idbOpen();
    await new Promise((res, rej) => {
      const tx = db.transaction("runs", "readwrite");
      tx.objectStore("runs").put(value, key);
      tx.oncomplete = res; tx.onerror = () => rej(tx.error);
    });
    db.close();
  }
  async function idbAll() {
    const db = await idbOpen();
    const out = await new Promise((res, rej) => {
      const tx = db.transaction("runs", "readonly");
      const req = tx.objectStore("runs").getAll();
      req.onsuccess = () => res(req.result); req.onerror = () => rej(req.error);
    });
    db.close();
    return out.map((s) => JSON.parse(s));
  }
  function download(obj, name) {
    const blob = new Blob([JSON.stringify(obj)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }
  async function exportAll() {
    const runs = await idbAll();
    download({ runs, exported_at: new Date().toISOString() }, `all_reels_${new Date().toISOString().slice(0, 10)}.json`);
    status = `EXPORTED ${runs.length} profiles: ${runs.map((r) => r.username).join(", ")}`;
    update();
  }

  document.querySelectorAll('script[type="application/json"]').forEach((s) => ingestText(s.textContent));
  update();

  function makePayload() {
    return { username, collected_at: new Date().toISOString(), page_url: location.href, status,
             n_items: items.size, items: [...items.values()] };
  }
  window.saveReels = function () {
    download(makePayload(), `${username}_reels_${new Date().toISOString().slice(0, 10)}.json`);
    console.log(`Saved ${items.size} items for @${username}.`);
  };
  autoBtn.onclick = () => autoRun();
  window.autoRun = autoRun;
  detailsBtn.onclick = getDetails;
  saveBtn.onclick = window.saveReels;
  exportBtn.onclick = exportAll;
  window.exportAll = exportAll;
  window.getDetails = getDetails;
  window.__reelsCollector = { items, update };
  console.log("Reels collector ready. Scroll the Reels tab, click Get details, wait for DETAILS DONE, then Save JSON.");
})();
