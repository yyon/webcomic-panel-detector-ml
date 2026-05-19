import { createCanvas, loadImage } from '@napi-rs/canvas';
import * as fs from "fs";
import * as path from "path";
import {getPanelsSimple, LoadScansForPanel, runModel, renderPages} from "./pageify.js";

const ratio = 2;
const minRatio = 2;
const maxRatio = 3;

const OffscreenCanvas = createCanvas;

async function saveCanvas(canvas, file) {
    const pngData = await canvas.encode('png');
    await fs.promises.writeFile(file, pngData)
}

/**
 * Handle a scan load, keeps original Image object to clone to insert scan somewhere
 */
class ScanLoader {
    constructor(path) {
        this.path = path;
        this.load();
    }
    /** Loads the scan */
    load() {
        this.loading = true
        this.error = false

        // The code below introduce a loading error for a quarter of the scans
        // think about updating the this.url at the bottom of the function and replace it with just url
        /*let url = this.url
        if (Math.floor(Math.random() * 4) === 2) {
            console.log("introduce error for scan url " + this.url)
            url = 'https://www.mangareader.net/fakeimage.jpg' //introduce an error 25% of the time
        }*/

        this.loadPromise = new Promise(async (resolve, reject) => {
            loadImage(this.path).then((img) => {
                this.scan = img;
                this.loading = false;
                this.error = false;
                resolve(img);                
            }).catch((err) => {
                console.error(err);
                this.error = true;
                reject(err);
            })
        });
        this.loadedWithoutError = this.loadPromise;

        return this.loadPromise
    }

    async findPanelsAsync() {
        if (this.panelsInfo != null) return this.panelsInfo

        const panels = findPanels(this.scan)
        const panel_at_top_edge = panels.length > 0 && panels[0][0] == 0
        const panel_at_bottom_edge =
            panels.length > 0 && panels[panels.length - 1][0] + panels[panels.length - 1][1] == this.scan.naturalHeight

        const panelsInfo = { panels, panel_at_top_edge, panel_at_bottom_edge }

        this.panelsInfo = panelsInfo
        return panelsInfo
    }
}

const processingType = process.argv[2];

const inFolder = process.argv[3];
console.log("folder", inFolder);  
if (inFolder == undefined) {
    throw new Error("no input folder");
}

const folders = [inFolder]
for (const subFolderName of fs.readdirSync(inFolder)) {
    const subFolder = path.join(inFolder, subFolderName);
    if (fs.lstatSync(subFolder).isDirectory()) {
        folders.push(subFolder);
    }
}


const outFolder = process.argv[4];
if (outFolder == undefined) {
    throw new Error("no output folder");
}
if (!fs.existsSync(outFolder)){
    fs.mkdirSync(outFolder);
}

for (const folder of folders) {
    const scans = [];

    const inputName = folder.replaceAll("/", "_");

    for (const fileName of fs.readdirSync(folder)) {
        const file = path.join(folder, fileName);
        if (fs.lstatSync(file).isDirectory()) continue;

        if (!fileName.endsWith(".xml")) {
            console.log(file);
            scans.push(new ScanLoader(file));
        }
    }

    if (processingType === "--parselongpanels") {
        const scansLoader = new LoadScansForPanel(scans);
        for await (const simplePanel of getPanelsSimple(scansLoader)) {
            console.log("panel", simplePanel);

            const panelWidth = simplePanel.scans[0].scan.naturalWidth
            const targetHeight = panelWidth * ratio
            console.log("simple panel", simplePanel, targetHeight, panelWidth)

            if (simplePanel.height > targetHeight) {
                console.log("long panel detected", simplePanel)

                const canvas = new OffscreenCanvas(panelWidth, simplePanel.height)
                const ctx = canvas.getContext("2d")

                let pos = 0
                for (const scan of scans) {
                    if (scan.loading) {
                        break
                    }
                    if (pos < simplePanel.y + simplePanel.height && pos + scan.scan.naturalHeight > simplePanel.y) {
                        ctx.drawImage(scan.scan, 0, pos - simplePanel.y)
                    }
                    pos += scan.scan.naturalHeight
                    if (pos >= simplePanel.y + simplePanel.height) {
                        break
                    }
                }

                console.log("running model")

                let results = await runModel(canvas)

                results = results
                    .map(box => [Math.round(box[0]), Math.round(box[1])])
                    .sort((a, b) => a[0] - b[0])
                    // .filter(box => box[1] > box[0])
                
                console.log("model results", results);

                const resultsWithHeight = results.map(box => [box[0], box[1] - box[0]])

                await saveCanvas(canvas, path.join(outFolder, `${inputName}_${simplePanel.y}.png`));
                fs.writeFileSync(path.join(outFolder, `${inputName}_${simplePanel.y}.json`), JSON.stringify(resultsWithHeight));
            }
        }
    } else if (processingType === "--pageify") {
        let pageNum = 1;
        for await (const page of renderPages(scans, minRatio, maxRatio / minRatio)) {
            await saveCanvas(page, path.join(outFolder, `${String(pageNum).padStart(4, "0")}.png`))
            pageNum++;
        }
    }

}
