import { createCanvas, loadImage } from '@napi-rs/canvas';
import * as ort from "onnxruntime-node";
import * as fs from "fs";
import * as path from "path";

const OffscreenCanvas = createCanvas;

const ratio = 2;

const modelURL = "../model.onnx";

async function saveCanvas(canvas, file) {
    const pngData = await canvas.encode('png');
    await fs.promises.writeFile(file, pngData)
}

const DEBUG_PANELS = false

const DEBUG_PANEL_WIDTH = 0.025
const DEBUG_PANEL_GAP = 0.01
const DEBUG_PANEL_NUM_LAYERS = 3

const SAVE_LONG_PANELS = true

let savedLongPanels = {}

let ort_cached = undefined
async function getOrtSession() {
    if (ort_cached === undefined) {
        ort_cached = await ort.InferenceSession.create(modelURL)
    }
    return ort_cached
}

const TARGET_WIDTH = 256
const WINDOW_HEIGHT = 1024
const STRIDE = 512
const SCORE_THRESHOLD = 0.5

// Resize using OffscreenCanvas
function resizeCanvas(source, targetWidth, targetHeight) {
    const resized = new OffscreenCanvas(targetWidth, targetHeight)
    const ctx = resized.getContext("2d")
    ctx.drawImage(source, 0, 0, targetWidth, targetHeight)
    return resized
}

// Extract vertical window with white fill if out of bounds
function cropWithWhiteFill(canvas, top, bottom) {
    const fullHeight = canvas.height
    const width = canvas.width
    const cropHeight = bottom - top

    const target = new OffscreenCanvas(width, cropHeight)
    const ctx = target.getContext("2d")

    ctx.fillStyle = "#fff"
    ctx.fillRect(0, 0, width, cropHeight)

    ctx.drawImage(canvas, 0, -top)
    return target
}

// Convert image to normalized Float32 tensor [1, 3, H, W]
function imageToTensor(canvas) {
    const ctx = canvas.getContext("2d")
    const { width, height } = canvas
    const imageData = ctx.getImageData(0, 0, width, height).data
    const data = new Float32Array(width * height * 3)
    console.log("image data", data);

    for (let x = 0; x < width; x++) {
        for (let y = 0; y < height; y++) {
            const inI = y*width + x
            const outI = y*width + x
            data[outI] = imageData[inI * 4] / 255 // R
            data[outI + width * height] = imageData[inI * 4 + 1] / 255 // G
            data[outI + width * height * 2] = imageData[inI * 4 + 2] / 255 // B
        }
    }

    return new ort.Tensor("float32", data, [1, 3, height, width])
}

function withinThreshold(a, b, threshold) {
    return Math.abs(a - b) < threshold
}

export async function runSlidingWindowInference(canvas, session) {
    const originalWidth = canvas.width
    const originalHeight = canvas.height

    const scale = originalWidth / TARGET_WIDTH
    const scaledWindowHeight = Math.round((WINDOW_HEIGHT * originalWidth) / TARGET_WIDTH)
    const stride = Math.round((STRIDE * originalWidth) / TARGET_WIDTH)

    const results = []

    const inputName = session.inputNames[0]
    const outputNames = session.outputNames

    for (let top = 0; top < originalHeight; top += stride) {
        const bottom = top + scaledWindowHeight
        console.log("sliding window", top, bottom)
        const cropped = cropWithWhiteFill(canvas, top, bottom)
        const resized = resizeCanvas(cropped, TARGET_WIDTH, WINDOW_HEIGHT)
        console.log("image size", resized);
        console.log("image", resized);

        const tensor = imageToTensor(resized)
        console.log("tensor", tensor);
        const output = await session.run({ [inputName]: tensor })

        const boxes = output[outputNames[0]].data
        const scores = output[outputNames[2]].data

        console.log("sliding window results", boxes, scores)

        for (let i = 0; i < scores.length; i++) {
            if (scores[i] >= SCORE_THRESHOLD) {
                const idx = i * 4
                let y1 = boxes[idx + 1] * scale + top
                if (y1 < 0) y1 = 0
                if (y1 > originalHeight) y1 = originalHeight
                let y2 = boxes[idx + 3] * scale + top
                if (y2 < 0) y1 = 0
                if (y2 > originalHeight) y2 = originalHeight
                console.log("box", boxes[idx+1], boxes[idx+3], y1, y2)
                if (y2 > y1) {
                    results.push({ y1, y2, score: scores[i], top, bottom })
                } else {
                    console.warn("0 height box", y1, y2, scores[i])
                }
            }
        }

        if (bottom >= originalHeight) break
    }

    // Postprocessing: remove redundant boxes
    const SAME_LOC_THRESHOLD = originalWidth * (12 / 256)
    const toRemove = new Set()

    for (let i = 0; i < results.length; i++) {
        for (let j = 0; j < results.length; j++) {
            if (i === j || toRemove.has(i) || toRemove.has(j)) continue

            const a = results[i]
            const b = results[j]

            let removeIdx = null

            if (a.y1 > b.y1 && a.y2 < b.y2) removeIdx = i
            else if (b.y1 > a.y1 && b.y2 < a.y2) removeIdx = j
            else if (a.top !== b.top) {
                const sameEdges = [
                    withinThreshold(a.y1, b.y1, SAME_LOC_THRESHOLD),
                    withinThreshold(a.y2, b.y2, SAME_LOC_THRESHOLD)
                ]
                const aHitWindow = [
                    withinThreshold(a.y1, a.top, SAME_LOC_THRESHOLD),
                    withinThreshold(a.y2, a.bottom, SAME_LOC_THRESHOLD)
                ]
                const bHitWindow = [
                    withinThreshold(b.y1, b.top, SAME_LOC_THRESHOLD),
                    withinThreshold(b.y2, b.bottom, SAME_LOC_THRESHOLD)
                ]

                if (sameEdges[0]) removeIdx = aHitWindow[1] ? i : bHitWindow[1] ? j : null
                else if (sameEdges[1]) removeIdx = aHitWindow[0] ? i : bHitWindow[0] ? j : null

                if (sameEdges[0] && sameEdges[1] && removeIdx === null) {
                    removeIdx = a.score > b.score ? j : i
                }
            }

            if (removeIdx !== null) toRemove.add(removeIdx)
        }
    }

    return results.filter((_, idx) => !toRemove.has(idx)).map(b => [b.y1, b.y2])
}

async function runModel(offscreenCanvas) {
    const session = await getOrtSession()
    console.log("loaded session")
    const boxes = await runSlidingWindowInference(offscreenCanvas, session)
    console.log("model results", boxes)
    return boxes
}

/**
 * @typedef Panel
 * @type {object}
 * @property {number} y
 * @property {number} height
 * @property {ScansLoader[]} scans
 * @property {string | undefined} panelType
 */

/**
 * @typedef Page
 * @type {object}
 * @property {number} y
 * @property {number} height
 * @property {{y: number, height: number}} panelBounds
 * @property {Panel[]} panels
 */

/**
 * @param {ScanLoader[]} scans
 * @returns {AsyncGenerator<ScanLoader>}
 */
async function* loadedScans(scans) {
    for (const scan of scans) {
        if (scan.loading && !scan.error) {
            await scan.loadPromise
        }
        if (scan.error) {
            throw new Error("scan has error - not handled yet")
        }
        yield scan
    }
}

/**
 * @param {ScanLoader[]} scans
 * @returns {AsyncGenerator<Panel>}
 */
async function* getPanelsSimple(scans) {
    let currentPos = 0

    /** @type {Panel | null} */
    let heldPanel = null

    let scanI = 0
    for await (const scan of loadedScans(scans)) {
        console.log("loaded scan", scan)

        const panelsInfo = await scan.findPanelsAsync()

        /** @type {Panel[]} */
        const panels = panelsInfo.panels.map(panel => ({
            y: currentPos + panel[0],
            height: panel[1],
            scans: [scan]
        }))

        if (heldPanel !== null) {
            if (panelsInfo.panel_at_top_edge && scan.scan.naturalWidth === heldPanel.scans[0].scan.naturalWidth) {
                // Merge heldPanel with the first panel of current scan
                panels[0] = {
                    y: heldPanel.y,
                    height: heldPanel.height + panels[0].height,
                    scans: heldPanel.scans.concat([scan])
                }
                heldPanel = null
            } else {
                console.log("created panel (held)", heldPanel, scanI)
                yield heldPanel
                heldPanel = null
            }
        }

        for (let i = 0; i < panels.length; i++) {
            const isLast = i === panels.length - 1
            if (isLast && panelsInfo.panel_at_bottom_edge) {
                heldPanel = panels[i] // Defer yielding in case we need to merge
            } else {
                console.log("created panel (normal)", panels[i], scanI)
                yield panels[i]
            }
        }

        currentPos += scan.scan.naturalHeight
        scanI++
    }

    // Emit any heldPanel that didn’t get merged
    console.log("created panel (held end)", heldPanel, scanI)
    if (heldPanel) yield heldPanel
}

/**
 * @param {ScanLoader[]} scans
 * @param {number} ratio
 * @returns {AsyncGenerator<Panel>}
 */
async function* getPanelsComplex(scans, ratio) {
    for await (const simplePanel of getPanelsSimple(scans)) {
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
                .filter(box => box[1] > box[0])

            if (SAVE_LONG_PANELS) {
                const saveName = simplePanel.scans[0].url + "_" + simplePanel.y + "_" + simplePanel.height
                savedLongPanels[saveName] = {
                    img: canvas,
                    results: results.map(box => [box[0], box[1] - box[0]])
                }
            }

            if (results.length === 0) {
                yield simplePanel
            } else {
                let currentPos = 0
                for (const box of results) {
                    const top = box[0]
                    const bottom = box[1]

                    if (top > currentPos) {
                        const whitespaceY = simplePanel.y + currentPos
                        const whitespaceHeight = top - currentPos
                        console.log("complex whitespace", whitespaceY, whitespaceHeight)
                        yield {
                            y: whitespaceY,
                            height: whitespaceHeight,
                            scans: simplePanel.scans,
                            panelType: "complexWhitespace"
                        }
                    }

                    const y = simplePanel.y + top
                    const height = bottom - top
                    console.log("complex panel", y, height)
                    const complexPanel = { y, height, scans: simplePanel.scans, panelType: "complex" }
                    yield complexPanel
                    currentPos = bottom
                }

                if (simplePanel.height > currentPos) {
                    const whitespaceY = simplePanel.y + currentPos
                    const whitespaceHeight = simplePanel.height - currentPos
                    console.log("complex whitespace end", whitespaceY, whitespaceHeight)
                    yield {
                        y: whitespaceY,
                        height: whitespaceHeight,
                        scans: simplePanel.scans,
                        panelType: "complexWhitespace"
                    }
                }
            }
        } else {
            yield simplePanel
        }
    }
}

/**
 *
 * @param {Panel[]} currentPagePanels
 * @param {Panel} nextPagePanel
 * @param {number} targetHeight
 * @returns {keepOnPage: Panel[], moveToNextPage: Panel[]}
 */
function checkBetterPageSplit(currentPagePanels, nextPagePanel, targetHeight) {
    const pageWidth = currentPagePanels[0].scans[0].scan.naturalWidth
    if (nextPagePanel.scans[0].scan.naturalWidth !== pageWidth) {
        return { keepOnPage: currentPagePanels, moveToNextPage: [] }
    }
    const lastPanelOnCurrentPage = currentPagePanels[currentPagePanels.length - 1]
    const whitespaces = [nextPagePanel.y - (lastPanelOnCurrentPage.y + lastPanelOnCurrentPage.height)]
    for (let i = currentPagePanels.length - 1; i > 0; i--) {
        const beforeSplitPanel = currentPagePanels[i - 1]
        const afterSplitPanel = currentPagePanels[i]
        const splitWhitespace = afterSplitPanel.y - (beforeSplitPanel.y + beforeSplitPanel.height)
        if (nextPagePanel.y + nextPagePanel.height - afterSplitPanel.y > targetHeight) {
            break
        }
        if (splitWhitespace > pageWidth * 0.25) {
            if (whitespaces.every(movedWhitespace => splitWhitespace > movedWhitespace * 2)) {
                return {
                    keepOnPage: currentPagePanels.slice(0, i),
                    moveToNextPage: currentPagePanels.slice(i)
                }
            }
        }
        whitespaces.push(splitWhitespace)
    }
    return { keepOnPage: currentPagePanels, moveToNextPage: [] }
}

/**
 * @param {ScanLoader[]} scans
 * @param {number} ratio
 * @returns {AsyncGenerator<Panel[]>}
 */
async function* groupPanelsIntoPages(scans, ratio) {
    /**@type Panel[] */
    let panelsBuffer = []

    for await (const panel of getPanelsComplex(scans, ratio)) {
        console.log("loaded panel", panel)

        const targetHeight = panel.scans[0].scan.naturalWidth * ratio

        if (panelsBuffer.length > 0) {
            const startY = panelsBuffer[0].y
            const endY = panel.y + panel.height
            if (
                endY - startY > targetHeight ||
                panelsBuffer[0].scans[0].scan.naturalWidth !== panel.scans[0].scan.naturalWidth
            ) {
                const betterSplit = checkBetterPageSplit(panelsBuffer, panel, targetHeight)
                yield betterSplit.keepOnPage
                panelsBuffer = betterSplit.moveToNextPage
            }
        }
        panelsBuffer.push(panel)
    }

    if (panelsBuffer.length > 0) {
        yield panelsBuffer
    }
}

/**
 * @param {ScanLoader[]} scans
 * @param {number} ratio
 * @returns {AsyncGenerator<Page>}
 */
async function* getPages(scans, ratio) {
    for await (const panels of await groupPanelsIntoPages(scans, ratio)) {
        console.log("loaded panel group", panels)

        let width = panels[0].scans[0].scan.naturalWidth
        let targetHeight = width * ratio
        let panelsTop = panels[0].y
        let panelsBottom = panels[panels.length - 1].y + panels[panels.length - 1].height
        let panelBounds = { y: panelsTop, height: panelsBottom - panelsTop }
        if (panelsBottom - panelsTop > targetHeight * 1.5) {
            for (let y = panelsTop; y + targetHeight < panelsBottom; y += Math.floor(targetHeight * 0.5)) {
                yield { y, height: targetHeight, panelBounds, panels }
            }
            yield { y: panelsBottom - targetHeight, height: targetHeight, panelBounds, panels }
        } else if (panelsBottom - panelsTop > targetHeight) {
            yield { y: panelsTop, height: panelsBottom - panelsTop, panelBounds, panels }
        } else {
            const center = Math.floor((panelsTop + panelsBottom) / 2)
            yield { y: Math.floor(center - targetHeight / 2), height: targetHeight, panelBounds, panels }
        }
    }
}

/**
 * @param {ScanLoader[]} scans
 * @param {Page} page
 * @param {Page | null} pageAbove
 * @param {Page | null} pageBelow
 * @param {Record<string, number>} urlCount
 * @returns {AsyncGenerator<ScanLoader>}
 */
async function renderPage(scans, page, pageAbove, pageBelow, urlCount) {
    console.log("rendering page", page, pageAbove, pageBelow)

    const scansWidth = page.panels[0].scans[0].scan.naturalWidth

    const debugPanelWidth = Math.round(DEBUG_PANEL_WIDTH * scansWidth)
    const debugPanelGap = Math.round(DEBUG_PANEL_GAP * scansWidth)

    const pageCanvas = new OffscreenCanvas(
        DEBUG_PANELS ? scansWidth + (debugPanelWidth + debugPanelGap) * DEBUG_PANEL_NUM_LAYERS : scansWidth,
        page.height
    )
    const pageCtx = pageCanvas.getContext("2d")

    const urls = []

    let pos = 0
    for await (const scan of loadedScans(scans)) {
        // console.log("loaded scan for page", scan)
        if (pos < page.y + page.height && pos + scan.scan.naturalHeight > page.y) {
            urls.push(scan.url)
            pageCtx.drawImage(scan.scan, 0, pos - page.y)
        }
        pos += scan.scan.naturalHeight
        if (pos >= page.y + page.height) {
            break
        }
    }
    console.log("loaded scans for page")

    if (DEBUG_PANELS) {
        let panelI = 0
        for (const pageForDebug of [pageAbove, page, pageBelow].filter(p => p != null)) {
            const opacity = pageForDebug === page ? 1 : 0.5
            for (const panel of pageForDebug.panels) {
                if (panel.panelType === "complexWhitespace") {
                    pageCtx.fillStyle = `rgba(255, 255, 255, ${opacity})`
                } else if (panel.panelType === "complex") {
                    pageCtx.fillStyle = `rgba(255, 0, 0, ${opacity})`
                } else {
                    pageCtx.fillStyle = `rgba(0, 255, 0, ${opacity})`
                }

                const posX =
                    scansWidth + debugPanelGap + (panelI % DEBUG_PANEL_NUM_LAYERS) * (debugPanelWidth + debugPanelGap)
                pageCtx.fillRect(posX, panel.y - page.y, debugPanelWidth, panel.height)
                panelI++
            }
        }
    }

    pageCtx.fillStyle = "rgba(0, 0, 0, 0.3)"

    if (pageAbove != null) {
        let panelAbove = pageAbove.panelBounds.y + pageAbove.panelBounds.height
        let panelAbovePosition = Math.min(page.panelBounds.y, panelAbove)
        if (panelAbovePosition > page.y) {
            pageCtx.fillRect(0, 0, scansWidth, panelAbovePosition - page.y)
        }
    }
    if (pageBelow != null) {
        let panelBelow = pageBelow.panelBounds.y
        let panelBelowPosition = Math.max(page.panelBounds.y + page.panelBounds.height, panelBelow)
        if (panelBelowPosition < page.y + page.height) {
            pageCtx.fillRect(0, panelBelowPosition - page.y, scansWidth, page.y + page.height - panelBelowPosition)
        }
    }

    urlCount[urls[0]] = (urlCount[urls[0]] ?? 0) + 1
    const representativeUrl = `${urls[0]}#${urlCount[urls[0]]}`

    const blob = await pageCanvas.convertToBlob()
    const blobURL = URL.createObjectURL(blob)
    const image = document.createElement("img")
    image.src = blobURL
    const s = new ScanLoader(representativeUrl, null)
    s.urls = urls
    s.loading = false
    s.scan = image
    s._original = null
    return s
}

/**
 * @param {ScanLoader[]} scans
 * @param {number} ratio
 * @returns {AsyncGenerator<ScanLoader>}
 */
async function* renderPages(scans, ratio) {
    let twoPagesAgo = null
    let onePageAgo = null
    let pageI = 0
    const urlCount = {}
    for await (const page of getPages(scans, ratio)) {
        console.log("loaded page", page, pageI)
        if (onePageAgo !== null) {
            yield renderPage(scans, onePageAgo, twoPagesAgo, page, urlCount)
        }
        twoPagesAgo = onePageAgo
        onePageAgo = page
        pageI++
    }
    if (onePageAgo !== null) {
        yield renderPage(scans, onePageAgo, twoPagesAgo, null, urlCount)
    }
}


function invertIntervals(intervals, totalLength) {
    const result = []
    let current = 0
    for (const [pos, size] of intervals) {
        if (pos > current) {
            result.push([current, pos - current])
        }
        current = pos + size
    }
    if (current < totalLength) {
        result.push([current, totalLength - current])
    }
    return result
}

function mergeSmallSegments(data, minSize, maxGap) {
    if (data.length === 1) return data
    let i = 0
    while (i < data.length) {
        const [pos, size] = data[i]
        if (size < minSize) {
            const prev = i > 0 ? data[i - 1] : null
            const next = i < data.length - 1 ? data[i + 1] : null

            const gapPrev = prev ? pos - (prev[0] + prev[1]) : Infinity
            const gapNext = next ? next[0] - (pos + size) : Infinity

            if (gapPrev <= gapNext && gapPrev <= maxGap) {
                const newStart = prev[0]
                const newEnd = pos + size
                data[i - 1] = [newStart, newEnd - newStart]
                data.splice(i, 1)
                i--
            } else if (gapNext <= maxGap) {
                const newStart = pos
                const newEnd = next[0] + next[1]
                data[i + 1] = [newStart, newEnd - newStart]
                data.splice(i, 1)
            } else {
                i++
            }
        } else {
            i++
        }
    }
    return data
}

function findPanels(img) {
    const width = Math.floor(img.naturalWidth * 0.95)
    const height = img.naturalHeight

    // Create canvas and draw original image
    const originalCanvas = new OffscreenCanvas(width, height)
    const ctxOriginal = originalCanvas.getContext("2d")
    ctxOriginal.drawImage(img, (width - img.naturalWidth) / 2, 0)

    // Step 1: Extract 1st column
    const colCanvas = new OffscreenCanvas(1, height)
    const ctxCol = colCanvas.getContext("2d")
    ctxCol.drawImage(originalCanvas, 0, 0, 1, height, 0, 0, 1, height)

    // Step 2: Stretch the column to full width
    const stretchedCanvas = new OffscreenCanvas(width, height)
    const ctxStretched = stretchedCanvas.getContext("2d")
    ctxStretched.imageSmoothingEnabled = false
    ctxStretched.drawImage(colCanvas, 0, 0, 1, height, 0, 0, width, height)

    // Step 3: Compute difference between original and stretched column
    const diffCanvas = new OffscreenCanvas(width, height)
    const ctxDiff = diffCanvas.getContext("2d")

    ctxDiff.drawImage(originalCanvas, 0, 0)
    ctxDiff.globalCompositeOperation = "difference"
    ctxDiff.drawImage(stretchedCanvas, 0, 0)

    // Step 4: Collapse horizontally using lighten to max RGB per row
    ctxDiff.globalCompositeOperation = "lighten"
    let crushWidth = width
    while (crushWidth > 1) {
        const split = Math.ceil(crushWidth / 2)
        ctxDiff.drawImage(diffCanvas, split, 0, crushWidth - split, height, 0, 0, crushWidth - split, height)
        crushWidth = split
    }

    const crushCanvas = new OffscreenCanvas(1, height)
    const ctxCrush = crushCanvas.getContext("2d")

    ctxCrush.drawImage(diffCanvas, 0, 0, 1, height, 0, 0, 1, height)

    // Step 5: Analyze vertical differences (1 column of height pixels)
    const diffColData = ctxCrush.getImageData(0, 0, 1, height).data
    const thresholds = []

    const colorTolerance = 5
    for (let y = 0; y < height; y++) {
        const i = y * 4
        const r = diffColData[i]
        const g = diffColData[i + 1]
        const b = diffColData[i + 2]
        const brightness = (r + g + b) / 3
        thresholds.push(brightness < colorTolerance) // true = white
    }

    // Step 6: Find runs of "true" (whitespace)
    let whitespaces = []
    let runStart = null
    for (let i = 0; i <= height; ++i) {
        const isWhite = thresholds[i] ?? false
        if (isWhite && runStart === null) {
            runStart = i
        } else if (!isWhite && runStart !== null) {
            const length = i - runStart
            whitespaces.push([runStart, length])
            runStart = null
        }
    }

    const minNonBWWhitespace = Math.floor(width / 10)
    const colData = ctxCol.getImageData(0, 0, 1, height).data
    whitespaces = whitespaces.filter(whitespace => {
        if (whitespace[1] < minNonBWWhitespace) {
            for (let sampleY = 0; sampleY < 1; sampleY += 0.1) {
                const y = Math.floor(whitespace[0] + whitespace[1] * sampleY)
                const i = y * 4
                const r = colData[i]
                const g = colData[i + 1]
                const b = colData[i + 2]
                if (Math.abs(r - g) > 10 || Math.abs(r - b) > 10) {
                    return false
                }
            }
        }
        return true
    })

    // Step 7: Invert whitespace into panel regions
    const panels = invertIntervals(whitespaces, height)

    const minPanel = Math.floor(width / 10)
    const maxGap = Math.floor(width / 40)
    const panelIsLine = Math.floor(width / 100)

    let mergedPanels = mergeSmallSegments(panels, minPanel, maxGap)
    mergedPanels = mergedPanels.filter(p => p[1] > panelIsLine)

    return mergedPanels
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

const inFolder = process.argv[2];
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

const inputName = inFolder.replaceAll("/", "_");

const outFolder = process.argv[3];
if (outFolder == undefined) {
    throw new Error("no output folder");
}
if (!fs.existsSync(outFolder)){
    fs.mkdirSync(outFolder);
}

for (const folder of folders) {
    const scans = [];

    for (const fileName of fs.readdirSync(folder)) {
        const file = path.join(folder, fileName);
        if (fs.lstatSync(file).isDirectory()) continue;

        if (!fileName.endsWith(".xml")) {
            console.log(file);
            scans.push(new ScanLoader(file));
        }
    }

    for await (const simplePanel of getPanelsSimple(scans)) {
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
}