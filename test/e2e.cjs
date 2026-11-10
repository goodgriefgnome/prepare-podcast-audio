// Drives the built app in Chromium: node test/e2e.cjs <input> <output.mp3> [stereo] [screenshotDir]
const { chromium } = require("playwright");
const fs = require("fs");
const os = require("os");
const path = require("path");
const [input, output, mode, shots] = process.argv.slice(2);
(async () => {
  // A persistent profile gets a normal storage quota (ephemeral contexts are capped).
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "ppa-profile-"));
  const browser = await chromium.launchPersistentContext(profile, { viewport: { width: 900, height: 1100 }, acceptDownloads: true });
  const page = browser.pages()[0] ?? (await browser.newPage());
  page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") console.log("[page]", m.text()); });
  page.on("pageerror", (e) => console.log("[pageerror]", e.message));
  await page.goto("http://localhost:4173/");
  console.log("storage quota GB:", await page.evaluate(async () => ((await navigator.storage.estimate()).quota / 1e9).toFixed(1)));
  if (shots) await page.screenshot({ path: `${shots}/1-empty.png` });
  await page.setInputFiles("#file", input);
  await page.waitForSelector("#view-ready:not([hidden])");
  console.log("meta:", await page.textContent("#ready-meta"), "| stereo disabled:", await page.isDisabled("#ch-stereo"));
  if (mode === "stereo") await page.click("#ch-stereo");
  if (shots) await page.screenshot({ path: `${shots}/2-ready.png`, fullPage: true });
  const t0 = Date.now();
  await page.click("#go");
  await page.waitForSelector("#view-proc:not([hidden])");
  let shot = false;
  const timer = setInterval(async () => {
    try {
      const pct = await page.textContent("#pct");
      console.log(`${((Date.now() - t0) / 1000).toFixed(0)}s ${pct} ${await page.textContent("#eta")}`);
      if (shots && !shot && parseInt(pct) > 40) { shot = true; await page.screenshot({ path: `${shots}/3-proc.png` }); }
    } catch {}
  }, 5000);
  await page.waitForSelector("#view-done:not([hidden]), #ready-error:not([hidden]), #empty-error:not([hidden])", { timeout: 4 * 3600 * 1000 });
  clearInterval(timer);
  if (!(await page.isVisible("#view-done"))) {
    console.log("FAILED:", await page.textContent("#ready-error"), await page.textContent("#empty-error"));
    process.exit(1);
  }
  console.log("done in", ((Date.now() - t0) / 1000).toFixed(1), "s:", (await page.textContent("#done-meta")), "|",
    await page.textContent("#stat-lufs"), "LUFS", await page.textContent("#stat-tp"), "dBTP |", await page.textContent("#stat-3"), await page.textContent("#stat-3-label"));
  if (shots) await page.screenshot({ path: `${shots}/4-done.png`, fullPage: true });
  const [dl] = await Promise.all([page.waitForEvent("download"), page.click("#save")]);
  console.log("download name:", dl.suggestedFilename());
  await dl.saveAs(output);
  await browser.close();
  fs.rmSync(profile, { recursive: true, force: true });
})();
