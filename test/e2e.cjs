// Drives the built app in Chromium: node test/e2e.cjs <input[,input2,...]> <outputDir-or-file.mp3> [stereo] [screenshotDir]
const { chromium } = require("playwright");
const fs = require("fs");
const os = require("os");
const path = require("path");
const [inputArg, output, mode, shots] = process.argv.slice(2);
const inputs = inputArg.split(",");
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
  await page.setInputFiles("#file", inputs);
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
  console.log("done in", ((Date.now() - t0) / 1000).toFixed(1), "s:", await page.textContent("#done-title"), "|", await page.textContent("#done-sub"));
  if (shots) await page.screenshot({ path: `${shots}/4-done.png`, fullPage: true });
  const cards = await page.$$("#results .result");
  for (const card of cards) {
    const t = async (k) => (await card.$eval(`[data-f="${k}"]`, (el) => (el.hidden ? "" : el.textContent))).trim();
    console.log(" -", await t("name"), "|", await t("pill"), "|", await t("meta"), "|", await t("lufs"), "LUFS", await t("tp"), "dBTP", (await t("miss")) || "");
    const save = await card.$('[data-f="save"]:not([hidden])');
    if (!save) continue;
    const [dl] = await Promise.all([page.waitForEvent("download"), save.click()]);
    const dest = output.endsWith(".mp3") && cards.length === 1 ? output : path.join(output, dl.suggestedFilename());
    if (!output.endsWith(".mp3")) fs.mkdirSync(output, { recursive: true });
    await dl.saveAs(dest);
    console.log("   saved", dest);
  }
  await browser.close();
  fs.rmSync(profile, { recursive: true, force: true });
})();
