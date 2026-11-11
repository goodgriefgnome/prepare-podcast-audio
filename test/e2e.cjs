// Drives the built app (npx vite preview --port 4173) in Chromium.
//
//   node test/e2e.cjs <input[,input2,...]> <outputDir> [stereo] [screenshotDir]
//
// Adds the files, waits for every row to finish, and saves each MP3.
const { chromium } = require("playwright");
const fs = require("fs");
const os = require("os");
const path = require("path");

const [inputArg, outDir, mode, shots] = process.argv.slice(2);
const inputs = inputArg.split(",");

(async () => {
  // A persistent profile gets a normal storage quota (ephemeral contexts are capped).
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "ppa-profile-"));
  const browser = await chromium.launchPersistentContext(profile, {
    viewport: { width: 1200, height: 1000 },
    acceptDownloads: true,
    args: ["--autoplay-policy=no-user-gesture-required"],
  });
  const page = browser.pages()[0] ?? (await browser.newPage());
  page.on("console", (m) => { if (m.type() === "error") console.log("[page]", m.text()); });
  page.on("pageerror", (e) => console.log("[pageerror]", e.message));
  await page.goto("http://localhost:4173/");
  if (mode === "stereo") await page.click("#ch-stereo");
  const t0 = Date.now();
  await page.setInputFiles("#file-input", inputs);
  const chips = () => page.$$eval("#files .chip", (els) => els.map((e) => e.textContent));
  let last = "";
  for (;;) {
    const c = await chips();
    const line = c.join(" | ");
    if (line !== last && Date.now() - t0 > 0) {
      console.log(`${((Date.now() - t0) / 1000).toFixed(0)}s ${line} · ${await page.textContent("#overall-text")}`);
      last = line;
    }
    if (c.length === inputs.length && c.every((x) => /Ready|Missed|Can’t read|Failed/.test(x))) break;
    await page.waitForTimeout(1000);
  }
  console.log("done in", ((Date.now() - t0) / 1000).toFixed(1), "s ·", await page.textContent("#overall-text"));
  if (shots) await page.screenshot({ path: `${shots}/done.png`, fullPage: true });
  fs.mkdirSync(outDir, { recursive: true });
  for (const row of await page.$$("#files .file")) {
    const t = async (k) => ((await row.$eval(`[data-f="${k}"]`, (el) => el.textContent)) || "").trim();
    console.log(" -", await t("name"), "|", await t("chip"), "|", await t("result"));
    const save = await row.$('[data-f="save"]:not([hidden])');
    if (!save) continue;
    const [dl] = await Promise.all([page.waitForEvent("download"), save.click()]);
    const dest = path.join(outDir, dl.suggestedFilename());
    await dl.saveAs(dest);
    console.log("   saved", dest);
  }
  await browser.close();
  fs.rmSync(profile, { recursive: true, force: true });
})();
