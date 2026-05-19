import { createCanvas, loadImage } from '@napi-rs/canvas';
import * as ort from "onnxruntime-node";

// pageify TODO
// make sure separate width scans are always separated
// better speech bubble direction detection
// improve algorithm
// performance

const OffscreenCanvas = createCanvas;

const modelURL = "../model.onnx";

const DEBUG_PANELS = false

const DEBUG_PANEL_WIDTH = 0.025
const DEBUG_PANEL_GAP = 0.01
const DEBUG_PANEL_NUM_LAYERS = 3

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

    for (let i = 0; i < width * height; i++) {
        data[i] = imageData[i * 4] / 255 // R
        data[i + width * height] = imageData[i * 4 + 1] / 255 // G
        data[i + width * height * 2] = imageData[i * 4 + 2] / 255 // B
    }

    return new ort.Tensor("float32", data, [1, 3, height, width])
}

/**
 * @typedef ModelResult
 * @type {object}
 * @property {number} y1
 * @property {number} y2
 * @property {number} x1
 * @property {number} x2
 * @property {number} score
 * @property {"complex" | "text" | "speech_up" | "speech_down"} panelType
 */

/**
 *
 * @param {number} pos
 * @returns {number}
 */
function normalDistributionProb(pos, scale) {
    return Math.pow(Math.E, -Math.pow(pos / scale, 2))
}

/**
 *
 * @param {ModelResult} panelOnAboveWindow
 * @param {number} bottomAboveWindow
 * @param {ModelResult} panelOnBelowWindow
 * @param {number} topBelowWindow
 * @returns {number} score
 */
function sameModelResultScore(resAbove, bottomAboveWindow, resBelow, topBelowWindow, imgWidth) {
    if (Math.min(resAbove.y2, resBelow.y2) < Math.max(resAbove.y1, resBelow.y1)) {
        console.log(
            "score",
            resAbove.name,
            resBelow.name,
            "not intersecting y",
            resAbove.y1,
            resAbove.y2,
            resBelow.y1,
            resBelow.y2
        )
        return 0
    }

    if ((resAbove.panelType === "complex") !== (resBelow.panelType === "complex")) {
        console.log("score", resAbove.name, resBelow.name, "panel types", resAbove.panelType, resBelow.panelType)
        return 0
    }

    if (resAbove.y2 < topBelowWindow) {
        console.log("score", resAbove.name, resBelow.name, "above window", resAbove.y2, topBelowWindow)
        return 0
    }

    if (resBelow.y1 > bottomAboveWindow) {
        console.log("score", resAbove.name, resBelow.name, "below window", resBelow.y1, bottomAboveWindow)
        return 0
    }

    const leftIntersection = Math.max(resAbove.x1, resBelow.x1)
    const rightIntersection = Math.min(resAbove.x2, resBelow.x2)
    const intersectionSize = rightIntersection - leftIntersection

    if (intersectionSize <= 0) {
        return 0
    }

    const xScore = Math.max(
        intersectionSize / (resAbove.x2 - resAbove.x1),
        intersectionSize / (resBelow.x2 - resBelow.x1)
    )

    const hitWindowScale = imgWidth * 0.15
    const aboveHitBottom =
        resBelow.y2 > resAbove.y2 ? normalDistributionProb(resAbove.y2 - bottomAboveWindow, hitWindowScale) : 0
    const belowHitTop =
        resAbove.y1 < resBelow.y1 ? normalDistributionProb(resBelow.y1 - topBelowWindow, hitWindowScale) : 0

    const samePosScale = imgWidth * 0.15
    const sameY1 = normalDistributionProb(resAbove.y1 - resBelow.y1, samePosScale)
    const sameY2 = normalDistributionProb(resAbove.y2 - resBelow.y2, samePosScale)

    const y1Score = sameY1 + belowHitTop - sameY1 * belowHitTop
    const y2Score = sameY2 + aboveHitBottom - sameY2 * aboveHitBottom

    const combinedScore = xScore * y1Score * y2Score
    console.log(
        "score",
        resAbove.name,
        resBelow.name,
        combinedScore,
        "(",
        xScore,
        sameY1,
        belowHitTop,
        sameY2,
        aboveHitBottom,
        ")"
    )
    return combinedScore
}

/**
 * @param {ModelResult} a
 * @param {ModelResult} b
 * @returns {ModelResult}
 */
function mergeModelResult(a, b) {
    return {
        ...a,
        ...b,
        x1: Math.min(a.x1, b.x1),
        x2: Math.max(a.x2, b.x2),
        y1: Math.min(a.y1, b.y1),
        y2: Math.max(a.y2, b.y2),
        score: Math.max(a.score, b.score),
        // TODO check cutoff
        modelType: a.score > b.score ? a.panelType : b.panelType,
        name: `m${a.name}|${b.name}`
    }
}

/**
 * @param {ModelResult[]} aboveWindowUnmatched
 * @param {number} bottomAboveWindow
 * @param {ModelResult[]} belowWindowUnmatched
 * @param {number} topBelowWindow
 * @param {{above: ModelResult, below: ModelResult, merged: ModelResult}[]} alreadyMatched
 * @param {number} imgWidth
 * @returns {{above: ModelResult, below: ModelResult, merged: ModelResult} | undefined} bestMatch
 */
function findModelResultMatch(
    aboveWindowUnmatched,
    bottomAboveWindow,
    belowWindowUnmatched,
    topBelowWindow,
    alreadyMatched,
    imgWidth
) {
    let bestMatch = undefined
    for (const resAbove of aboveWindowUnmatched) {
        for (const resBelow of belowWindowUnmatched) {
            const matchScore = sameModelResultScore(resAbove, bottomAboveWindow, resBelow, topBelowWindow, imgWidth)

            if (bestMatch === undefined || matchScore > bestMatch.score) {
                bestMatch = {
                    above: resAbove,
                    below: resBelow,
                    score: matchScore
                }
            }
        }
    }

    if (bestMatch === undefined) {
        console.log("merged all")
        return undefined
    }

    if (bestMatch.score < 0.5) {
        console.log("below threshold", bestMatch.score)
        return undefined
    }

    if (
        alreadyMatched.some(
            already =>
                sameModelResultScore(bestMatch.above, bottomAboveWindow, already.below, topBelowWindow, imgWidth) >
                bestMatch.score
        )
    ) {
        console.log("better existing match below")
        return undefined
    }
    if (
        alreadyMatched.some(
            already =>
                sameModelResultScore(already.above, bottomAboveWindow, bestMatch.below, topBelowWindow, imgWidth) >
                bestMatch.score
        )
    ) {
        console.log("better existing match above")
        return undefined
    }

    console.log("found match", bestMatch.score)
    return {
        above: bestMatch.above,
        below: bestMatch.below,
        merged: mergeModelResult(bestMatch.above, bestMatch.below),
        score: bestMatch.score
    }
}

function removeModelResultsInsideOtherResults(results, tolerance) {
    const startTime = performance.now()

    results = [...results]

    const toRemove = new Set()

    for (let i = 0; i < results.length; i++) {
        for (let j = 0; j < results.length; j++) {
            if (i === j || toRemove.has(i) || toRemove.has(j)) continue

            const a = results[i]
            const b = results[j]

            if ((a.panelType === "complex") !== (b.panelType === "complex")) continue

            if (a.x1 > b.x2 || a.x2 < b.x1) continue

            let removeIdx = null

            if (a.y1 + tolerance > b.y1 && a.y2 < b.y2 + tolerance) {
                results[j] = mergeModelResult(a, b)
                console.log("merged panel", a.name, "inside", b.name, a, b, "-", results[j])
                removeIdx = i
            } else if (b.y1 + tolerance > a.y1 && b.y2 < a.y2 + tolerance) {
                results[i] = mergeModelResult(a, b)
                console.log("merged panel", b.name, "inside", a.name, b, a, "-", results[i])
                removeIdx = j
            }

            if (removeIdx !== null) toRemove.add(removeIdx)
        }
    }

    const endTime = performance.now()
    console.log(`time removeModelResultsInsideOtherResults ${endTime - startTime}`)

    return results.filter((_, idx) => !toRemove.has(idx))
}

let modelRunI = 0
export async function runSlidingWindowInference(canvas, session) {
    modelRunI++
    const originalWidth = canvas.width
    const originalHeight = canvas.height

    const scale = originalWidth / TARGET_WIDTH
    const scaledWindowHeight = Math.round((WINDOW_HEIGHT * originalWidth) / TARGET_WIDTH)
    const stride = Math.round((STRIDE * originalWidth) / TARGET_WIDTH)

    let originalResults = []
    const resultsWindows = []

    const inputName = session.inputNames[0]
    const outputNames = session.outputNames
    let resI = 0

    const insideTolerance = originalWidth * (12 / 256)

    for (let top = 0; top < originalHeight; top += stride) {
        const bottom = top + scaledWindowHeight
        console.log("sliding window", top, bottom)
        const cropped = cropWithWhiteFill(canvas, top, bottom)
        const resized = resizeCanvas(cropped, TARGET_WIDTH, WINDOW_HEIGHT)

        const tensor = imageToTensor(resized)
        let output
        try {
            output = await session.run({ [inputName]: tensor })
        } catch (e) {
            console.error("error running model", e)
            console.log("tensor", tensor)
            output = {
                [outputNames[0]]: { data: [] },
                [outputNames[1]]: { data: [] },
                [outputNames[2]]: { data: [] }
            }
        }
        tensor.dispose()

        const boxes = output[outputNames[0]].data
        const labels = output[outputNames[1]].data
        const scores = output[outputNames[2]].data

        console.log("sliding window results", boxes, labels, scores)

        const resultsWindow = []
        for (let i = 0; i < scores.length; i++) {
            if (scores[i] >= SCORE_THRESHOLD) {
                const idx = i * 4
                let y1 = boxes[idx + 1] * scale + top
                if (y1 < 0) y1 = 0
                if (y1 > originalHeight) y1 = originalHeight
                let y2 = boxes[idx + 3] * scale + top
                if (y2 < 0) y1 = 0
                if (y2 > originalHeight) y2 = originalHeight
                const x1 = boxes[idx] * scale
                const x2 = boxes[idx + 2] * scale
                if (y2 > y1) {
                    resI++
                    resultsWindow.push({
                        y1,
                        y2,
                        x1,
                        x2,
                        score: scores[i],
                        top,
                        bottom,
                        panelType: ["complex", "text", "speech_up", "speech_down"][Number(labels[i]) - 1],
                        windowI: top / stride,
                        windowTop: top,
                        windowBottom: bottom,
                        name: `${modelRunI}-${top / stride}-${resI}`
                    })
                } else {
                    console.warn("0 height box", y1, y2, scores[i])
                }
            }
        }
        originalResults = originalResults.concat(resultsWindow)
        resultsWindows.push(removeModelResultsInsideOtherResults(resultsWindow, insideTolerance))

        if (bottom >= originalHeight) break
    }

    // Postprocessing: remove redundant boxes
    let resultsMerged = [...resultsWindows[0]]
    let bottomAboveWindow = scaledWindowHeight

    for (let windowI = 1; windowI < resultsWindows.length; windowI++) {
        const startTime = performance.now()
        console.log("merging sliding window")
        const topBelowWindow = stride * windowI
        const bottomBelowWindow = topBelowWindow + scaledWindowHeight

        let remainingResultsAbove = [...resultsMerged]
        let remainingResultsBelow = [...resultsWindows[windowI]]
        const merged = []

        let bestMerge
        while (
            (bestMerge = findModelResultMatch(
                remainingResultsAbove,
                bottomAboveWindow,
                remainingResultsBelow,
                topBelowWindow,
                merged,
                originalWidth
            )) !== undefined
        ) {
            remainingResultsAbove = remainingResultsAbove.filter(resAbove => resAbove !== bestMerge.above)
            remainingResultsBelow = remainingResultsBelow.filter(resBelow => resBelow !== bestMerge.below)
            console.log(
                "merged panel sliding",
                bestMerge.above.name,
                bestMerge.below.name,
                bestMerge.above,
                bestMerge.below,
                "-",
                bestMerge.merged
            )
            merged.push(bestMerge)
        }

        console.log(
            "merged sliding window",
            merged,
            remainingResultsAbove,
            remainingResultsBelow,
            windowI,
            bottomAboveWindow,
            topBelowWindow,
            resultsMerged,
            resultsWindows[windowI]
        )

        resultsMerged = [
            ...remainingResultsAbove,
            ...remainingResultsBelow,
            ...merged.map(res => {
                return { ...res.merged, mergedRes: [...(res.mergedRes ?? []), res.above, res.below] }
            })
        ]

        bottomAboveWindow = bottomBelowWindow
        const endTime = performance.now()
        console.log(`time merge sliding window (${merged.length}) ${endTime - startTime}`)
    }

    return [removeModelResultsInsideOtherResults(resultsMerged, insideTolerance), originalResults]
}

export async function runModel(offscreenCanvas) {
    const startTime = performance.now()

    const session = await getOrtSession()
    console.log("loaded session")
    const boxes = await runSlidingWindowInference(offscreenCanvas, session)
    console.log("model results", boxes)

    const endTime = performance.now()
    console.log(`time runModel ${endTime - startTime}`)

    return boxes
}

class ScanForPanels {
    constructor(scan, scanY) {
        this.scanY = scanY

        this.scanLoader = scan
    }

    async getCleanImage() {
        await this.scanLoader.loadPromise
        this.scan = this.scanLoader.scan
    }

    findPanelsCached() {
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

/**
 * @typedef Panel
 * @type {object}
 * @property {number} y
 * @property {number} height
 * @property {ScanForPanels[]} scans
 * @property {number} scansStartY
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

export class LoadScansForPanel {
    scansForPanel = {}
    scanHeights = {}

    constructor(originalScans) {
        this.originalScans = originalScans
    }

    numScans() {
        return this.originalScans.length
    }

    async getScanTop(scanI) {
        let top = 0
        for (let i = 0; i < scanI; i++) {
            const scanHeight = await this.getScanHeight(i)
            top += scanHeight
        }
        return top
    }

    async getScanHeight(i) {
        if (this.scanHeights[i] === undefined) {
            await this.getLoadedScans(i)
        }
        return this.scanHeights[i]
    }

    async getLoadedScans(scanI) {
        const scan = this.originalScans[scanI]

        if (this.scansForPanel[scanI] === undefined) {
            await scan.loadedWithoutError.promise
            const scanTop = await this.getScanTop(scanI)
            const scanForPanel = new ScanForPanels(scan, scanTop)
            await scanForPanel.getCleanImage()
            this.scanHeights[scanI] = scanForPanel.scan.naturalHeight
            this.scansForPanel[scanI] = scanForPanel
            console.log("loaded scan", scanI, this.scanHeights[scanI], scanTop)
        }

        return this.scansForPanel[scanI]
    }

    removeLoadedScanFromCache(i) {
        this.scansForPanel[i] = undefined
    }
}

/**
 * @param {LoadScansForPanel} scans
 * @returns {AsyncGenerator<Panel>}
 */
export async function* getPanelsSimple(scans) {
    let currentPos = 0

    /** @type {Panel | null} */
    let heldPanel = null

    for (let scanI = 0; scanI < scans.numScans(); scanI++) {
        const scan = await scans.getLoadedScans(scanI)
        console.log("loaded scan", scan)

        const panelsInfo = scan.findPanelsCached()

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
    }

    // Emit any heldPanel that didn’t get merged
    console.log("created panel (held end)", heldPanel)
    if (heldPanel) yield heldPanel
}

/**
 * @param {LoadScansForPanel} scans
 * @param {number} ratio
 * @returns {AsyncGenerator<Panel>}
 */
async function* getPanelsComplex(scans, ratio, stretch) {
    for await (const simplePanel of getPanelsSimple(scans)) {
        const panelWidth = simplePanel.scans[0].scan.naturalWidth
        const targetHeight = panelWidth * ratio
        console.log("simple panel", simplePanel, targetHeight, panelWidth)

        if (simplePanel.height > targetHeight) {
            console.log("long panel detected", simplePanel)

            const canvas = new OffscreenCanvas(panelWidth, simplePanel.height)
            const ctx = canvas.getContext("2d")

            let pos = simplePanel.scans[0].scanY
            for (const scan of simplePanel.scans) {
                if (pos < simplePanel.y + simplePanel.height && pos + scan.scan.naturalHeight > simplePanel.y) {
                    ctx.drawImage(scan.scan, 0, pos - simplePanel.y)
                }
                pos += scan.scan.naturalHeight
                if (pos >= simplePanel.y + simplePanel.height) {
                    break
                }
            }

            console.log("running model")

            const [model_results_merged, model_results_unmerged] = await runModel(canvas)
            console.log("model results", model_results_merged)

            const results_unfiltered = model_results_merged
                .map(box => {
                    return {
                        y1: Math.round(box.y1),
                        y2: Math.round(box.y2),
                        x1: Math.round(box.x1),
                        x2: Math.round(box.x2),
                        panelType: box.panelType,
                        name: box.name
                    }
                })
                .sort((a, b) => a.y1 - b.y1)
                .filter(box => box.y2 > box.y1)

            const toRemove = new Set()

            for (let i = 0; i < results_unfiltered.length; i++) {
                for (let j = 0; j < results_unfiltered.length; j++) {
                    if (i === j || toRemove.has(i) || toRemove.has(j)) continue

                    const a = results_unfiltered[i]
                    const b = results_unfiltered[j]

                    let removeIdx = null

                    if (a.y1 > b.y1 && a.y2 < b.y2) removeIdx = i
                    else if (b.y1 > a.y1 && b.y2 < a.y2) removeIdx = j

                    if (removeIdx !== null) toRemove.add(removeIdx)
                }
            }

            let results = results_unfiltered.filter((_, idx) => !toRemove.has(idx)).sort((a, b) => a.y1 - b.y1)

            for (let boxI = 0; boxI < results.length; boxI++) {
                let box = results[boxI]
                if (box.panelType === "speech_up" || box.panelType == "speech_down" || box.panelType === "text") {
                    let prevOverlap = 0
                    if (boxI !== 0) {
                        const prevBox = results[boxI - 1]
                        if (prevBox.panelType === "complex") {
                            if (box.y2 - prevBox.y1 < targetHeight * stretch) {
                                prevOverlap = (prevBox.y2 + panelWidth * 0.1 - box.y1) / (box.y2 - box.y1)
                            }
                        }
                    }
                    let nextOverlap = 0
                    if (boxI !== results.length - 1) {
                        const nextBox = results[boxI + 1]
                        let nextPanelI = boxI + 1
                        while (true) {
                            const nextPanel = results[nextPanelI]
                            if (
                                !(
                                    nextPanel.panelType === "speech_up" ||
                                    nextPanel.panelType == "speech_down" ||
                                    nextPanel.panelType === "text"
                                )
                            ) {
                                break
                            }
                            if (nextPanelI === results.length - 1) {
                                break
                            }
                            nextPanelI++
                        }
                        const nextPanel = results[nextPanelI]
                        if (nextPanel.panelType === "complex") {
                            if (nextPanel.y2 - box.y1 < targetHeight * stretch) {
                                nextOverlap = (box.y2 - (nextBox.y1 - panelWidth * 0.1)) / (box.y2 - box.y1)
                            }
                        }
                    }

                    if (prevOverlap > 0 || nextOverlap > 0) {
                        let mergeNext
                        if (prevOverlap === 0 || nextOverlap === 0) {
                            mergeNext = nextOverlap > prevOverlap
                        } else if (box.panelType === "speech_up") {
                            mergeNext = false
                        } else if (box.panelType === "speech_down") {
                            mergeNext = true
                        } else {
                            mergeNext = nextOverlap > prevOverlap
                        }

                        if (mergeNext) {
                            results[boxI + 1].y1 = box.y1
                        } else {
                            results[boxI - 1].y2 = box.y2
                        }
                        results.splice(boxI, 1)
                        boxI--
                    }
                }
            }

            const original_results = [
                model_results_merged.map(panel => {
                    return {
                        ...panel,
                        y: simplePanel.y + panel.y1,
                        height: panel.y2 - panel.y1,
                        windowTop: simplePanel.y + (panel.windowTop ?? 0),
                        windowBottom: simplePanel.y + (panel.windowBottom ?? 0)
                    }
                }),
                model_results_unmerged.map(panel => {
                    return {
                        ...panel,
                        y: simplePanel.y + panel.y1,
                        height: panel.y2 - panel.y1,
                        windowTop: simplePanel.y + (panel.windowTop ?? 0),
                        windowBottom: simplePanel.y + (panel.windowBottom ?? 0)
                    }
                })
            ]

            if (results.length === 0) {
                yield simplePanel
            } else {
                if (results[0].y1 > 0 && results[0].y1 < panelWidth * 0.08 && results[0].y2 < targetHeight * stretch) {
                    results[0].y1 = 0
                }
                if (
                    results[results.length - 1].y2 < simplePanel.height &&
                    simplePanel.height - results[results.length - 1].y2 < panelWidth * 0.08 &&
                    simplePanel.height - results[results.length - 1].y1 < targetHeight * stretch
                ) {
                    results[results.length - 1].y2 = simplePanel.height
                }

                let currentPos = 0
                for (const box of results) {
                    const top = box.y1
                    const bottom = box.y2
                    const panelType = box.panelType

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
                    console.log("complex panel", y, height, panelType)
                    const complexPanel = {
                        y,
                        height,
                        scans: simplePanel.scans,
                        panelType,
                        x1: box.x1,
                        x2: box.x2,
                        original_results: DEBUG_PANELS ? original_results : undefined
                    }
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
 * @param {LoadScansForPanel} scans
 * @param {number} ratio
 * @returns {AsyncGenerator<Panel[]>}
 */
async function* groupPanelsIntoPages(scans, ratio, stretch) {
    /**@type Panel[] */
    let panelsBuffer = []

    let pageI = 0
    for await (const panel of getPanelsComplex(scans, ratio, stretch)) {
        console.log("loaded panel", panel)

        const targetHeight = panel.scans[0].scan.naturalWidth * ratio

        if (panelsBuffer.length > 0) {
            const startY = panelsBuffer[0].y
            const prevPageEndY = panelsBuffer[panelsBuffer.length - 1].y + panelsBuffer[panelsBuffer.length - 1].height
            const endY = panel.y + panel.height
            const tooMuchOverlap = (prevPageEndY - panel.y) / panel.height > 0.9
            const combineWithSuperLongPanel =
                (prevPageEndY - startY > targetHeight * stretch &&
                    ((prevPageEndY - panel.y) / panel.height > 0.75 || panel.panelType === "complexWhitespace")) ||
                ((panel.height > targetHeight) & stretch &&
                    (prevPageEndY - panel.y) / (prevPageEndY - startY) > 0.75 &&
                    panelsBuffer.every(panel => panel.panelType !== undefined))
            console.log(
                "overlap",
                pageI,
                (prevPageEndY - panel.y) / panel.height,
                endY - startY > targetHeight,
                endY - startY > (tooMuchOverlap ? targetHeight * stretch : targetHeight),
                startY,
                prevPageEndY,
                panel.y,
                endY
            )
            if (
                (endY - startY > (tooMuchOverlap ? targetHeight * stretch : targetHeight) &&
                    !combineWithSuperLongPanel) ||
                panelsBuffer[0].scans[0].scan.naturalWidth !== panel.scans[0].scan.naturalWidth
            ) {
                const betterSplit = checkBetterPageSplit(panelsBuffer, panel, targetHeight)
                console.log("betterSplit", pageI, betterSplit)
                yield betterSplit.keepOnPage
                pageI++
                panelsBuffer = betterSplit.moveToNextPage
            }
        }
        panelsBuffer.push({ ...panel, pageNum: pageI })
    }

    if (panelsBuffer.length > 0) {
        yield panelsBuffer
    }
}

function panelGroupStart(panelGroup) {
    return panelGroup[0].y
}
function panelGroupEnd(panelGroup) {
    return panelGroup[panelGroup.length - 1].y + panelGroup[panelGroup.length - 1].height
}
function panelGroupHeight(panelGroup) {
    return panelGroupEnd(panelGroup) - panelGroupStart(panelGroup)
}

/**
 * @param {LoadScansForPanel} scans
 * @param {number} ratio
 * @returns {AsyncGenerator<Page>}
 */
async function* groupPanelsIntoPagesRemoveWhitespace(scans, ratio, stretch) {
    let lastPage = null
    let whitespacePage = null
    for await (let page of groupPanelsIntoPages(scans, ratio, stretch)) {
        if (page.every(panel => panel.panelType === "complexWhitespace")) {
            whitespacePage = page
            continue
        }

        if (whitespacePage != null) {
            const allPanels = [lastPage ?? [], whitespacePage, page].flat()
            const scansWidth = Math.max(...allPanels.map(panel => panel.scans[0].scan.naturalWidth))
            const targetHeight = scansWidth * ratio
            const stretchHeight = targetHeight * stretch

            if (lastPage == null) {
                const combinedPage = whitespacePage.concat(page)
                if (panelGroupHeight(combinedPage) > stretchHeight) {
                    yield whitespacePage
                    lastPage = page
                    whitespacePage = null
                    continue
                }
                lastPage = combinedPage
                whitespacePage = null
                continue
            }

            const splitPointAbs = Math.min(
                Math.max(Math.round((panelGroupStart(lastPage) + panelGroupEnd(page)) / 2), panelGroupEnd(lastPage)),
                panelGroupStart(page)
            )

            const whitespacePageScans = []
            for (const whitespaceScan of whitespacePage) {
                for (const scan of whitespaceScan.scans) {
                    if (!whitespacePageScans.includes(scan)) {
                        whitespacePageScans.push(scan)
                    }
                }
            }
            const splitEvenly = {
                before:
                    splitPointAbs === panelGroupEnd(lastPage)
                        ? lastPage
                        : lastPage.concat([
                              {
                                  y: panelGroupStart(whitespacePage),
                                  height: splitPointAbs - panelGroupStart(whitespacePage),
                                  panelType: "complexWhitespace",
                                  scans: whitespacePageScans
                              }
                          ]),
                after:
                    splitPointAbs === panelGroupStart(page)
                        ? page
                        : [
                              {
                                  y: splitPointAbs,
                                  height: panelGroupEnd(whitespacePage) - splitPointAbs,
                                  panelType: "complexWhitespace",
                                  scans: whitespacePageScans
                              }
                          ].concat(page)
            }

            let splitOptions = [splitEvenly]
            for (let i = 0; i <= whitespacePage.length; i++) {
                splitOptions.push({
                    before: lastPage.concat(whitespacePage.slice(0, i)),
                    after: whitespacePage.slice(i).concat(page)
                })
            }

            console.log(
                "options",
                splitOptions.map(option => {
                    return {
                        beforeHeight: panelGroupHeight(option.before),
                        afterHeight: panelGroupHeight(option.after),
                        valid:
                            panelGroupHeight(option.before) < stretchHeight &&
                            panelGroupHeight(option.after) < stretchHeight,
                        option
                    }
                }),
                "whitespace",
                whitespacePage
            )

            splitOptions = splitOptions.filter(
                ({ before, after }) =>
                    panelGroupHeight(before) < stretchHeight && panelGroupHeight(after) < stretchHeight
            )

            if (splitOptions.length > 0) {
                const bestOption = splitOptions
                    .sort(
                        (a, b) =>
                            panelGroupHeight(a.before) +
                            panelGroupHeight(a.after) -
                            (panelGroupHeight(b.before) + panelGroupHeight(b.after))
                    )
                    .sort(
                        (a, b) =>
                            Math.max(panelGroupHeight(a.before), panelGroupHeight(a.after)) -
                            Math.max(panelGroupHeight(b.before), panelGroupHeight(b.after))
                    )[0]

                console.log("remove whitespace", bestOption)

                yield bestOption.before
                lastPage = bestOption.after
                whitespacePage = null
                continue
            } else {
                yield lastPage
                yield whitespacePage
                lastPage = page
                whitespacePage = null
                continue
            }
        }

        if (lastPage !== null) {
            yield lastPage
        }
        lastPage = page
    }
    if (lastPage !== null) {
        yield lastPage
    }
    if (whitespacePage !== null) {
        yield whitespacePage
    }
}

/**
 * @param {LoadScansForPanel} scans
 * @param {number} ratio
 * @returns {AsyncGenerator<Page>}
 */
async function* getPages(scans, ratio, stretch) {
    for await (const panels of await groupPanelsIntoPagesRemoveWhitespace(scans, ratio, stretch)) {
        console.log("loaded panel group", panels)

        let width = panels[0].scans[0].scan.naturalWidth
        let targetHeight = width * ratio
        let panelsTop = panels[0].y
        let panelsBottom = panels[panels.length - 1].y + panels[panels.length - 1].height
        let panelBounds = { y: panelsTop, height: panelsBottom - panelsTop }
        console.log("page size", panels[0].pageNum, (panelsBottom - panelsTop) / targetHeight)
        if (panelsBottom - panelsTop > targetHeight * stretch) {
            const numPanPages =
                1 + Math.max(1, Math.round((panelsBottom - panelsTop - targetHeight) / (targetHeight * 0.5)))
            const panAmt = (panelsBottom - panelsTop - targetHeight) / (numPanPages - 1)
            for (let panI = 0; panI < numPanPages; panI++) {
                let y = Math.floor(panelsTop + panAmt * panI)
                yield {
                    y,
                    height: targetHeight,
                    panelBounds,
                    panels,
                    panI,
                    fadeTop: panI !== 0,
                    fadeBottom: panI !== numPanPages - 1
                }
            }
        } else if (panelsBottom - panelsTop > targetHeight) {
            yield { y: panelsTop, height: panelsBottom - panelsTop, panelBounds, panels }
        } else {
            const center = Math.floor((panelsTop + panelsBottom) / 2)
            yield { y: Math.floor(center - targetHeight / 2), height: targetHeight, panelBounds, panels }
        }
    }
}

/**
 * @param {LoadScansForPanel} scans
 * @param {Page} page
 * @param {Page | null} pageAbove
 * @param {Page | null} pageBelow
 * @param {Record<string, number>} urlCount
 * @returns {AsyncGenerator<ScanLoader>}
 */
export async function renderPage(scans, page, pageAbove, pageBelow, urlCount) {
    const startTime = performance.now()

    console.log("rendering page", page, pageAbove, pageBelow)

    const scansWidth = page.panels[0].scans[0].scan.naturalWidth

    const debugPanelWidth = Math.round(DEBUG_PANEL_WIDTH * scansWidth)
    const debugPanelGap = Math.round(DEBUG_PANEL_GAP * scansWidth)

    const pageCanvas = new OffscreenCanvas(DEBUG_PANELS
        ? scansWidth + (debugPanelWidth + debugPanelGap) * DEBUG_PANEL_NUM_LAYERS
        : scansWidth, page.height)
    const pageCtx = pageCanvas.getContext("2d")

    const urls = []

    let scansCompleted = -1
    for (let scanI = 0; scanI < scans.numScans(); scanI++) {
        const pos = await scans.getScanTop(scanI)
        const height = await scans.getScanHeight(scanI)
        if (pos < page.y + page.height && pos + height > page.y) {
            const scan = await scans.getLoadedScans(scanI)
            urls.push(scan.url)
            pageCtx.drawImage(scan.scan, 0, pos - page.y)
        } else if (pos >= page.y + page.height) {
            break
        } else if (pos + height < page.y) {
            scans.removeLoadedScanFromCache(scanI)
            scansCompleted = scanI
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
                } else if (panel.panelType === "text") {
                    pageCtx.fillStyle = `rgba(0, 0, 255, ${opacity})`
                } else if (panel.panelType === "speech_up") {
                    pageCtx.fillStyle = `rgba(255, 0, 255, ${opacity})`
                } else if (panel.panelType === "speech_down") {
                    pageCtx.fillStyle = `rgba(150, 0, 255, ${opacity})`
                } else {
                    pageCtx.fillStyle = `rgba(0, 255, 0, ${opacity})`
                }

                const posX =
                    scansWidth + debugPanelGap + (panelI % DEBUG_PANEL_NUM_LAYERS) * (debugPanelWidth + debugPanelGap)
                pageCtx.fillRect(posX, panel.y - page.y, debugPanelWidth, panel.height)
                panelI++
            }
        }

        console.log("page", page.panels)
        const modelResultsMerged = new Set(page.panels.map(panel => panel.original_results?.[0] ?? []).flat())
        const modelResultsUnmerged = new Set(page.panels.map(panel => panel.original_results?.[1] ?? []).flat())
        console.log("model results", modelResultsMerged)
        for (const modelResult of modelResultsMerged) {
            pageCtx.lineWidth = Math.floor(scansWidth * 0.015)
            pageCtx.strokeStyle = "black"
            pageCtx.strokeRect(
                modelResult.x1 ?? 0,
                modelResult.y - page.y,
                (modelResult.x2 ?? scansWidth) - (modelResult.x1 ?? 0),
                modelResult.height
            )
            pageCtx.lineWidth = Math.floor(scansWidth * 0.01)
            if (modelResult.panelType === "complex") {
                pageCtx.strokeStyle = `rgba(255, 0, 0, 1)`
            } else if (modelResult.panelType === "text") {
                pageCtx.strokeStyle = `rgba(0, 0, 255, 1)`
            } else if (modelResult.panelType === "speech_up") {
                pageCtx.strokeStyle = `rgba(255, 0, 255, 1)`
            } else if (modelResult.panelType === "speech_down") {
                pageCtx.strokeStyle = `rgba(150, 0, 255, 1)`
            }
            pageCtx.strokeRect(
                modelResult.x1 ?? 0,
                modelResult.y - page.y,
                (modelResult.x2 ?? scansWidth) - (modelResult.x1 ?? 0),
                modelResult.height
            )
            if (modelResult.windowI !== undefined && modelResult.name !== undefined) {
                pageCtx.fillStyle = "black"
                pageCtx.fillRect(
                    modelResult.x1 ?? 0 + 16 + 128 * modelResult.windowI,
                    modelResult.y - page.y + 16,
                    128,
                    32
                )
                pageCtx.fillStyle = pageCtx.strokeStyle
                pageCtx.font = "32px serif"
                pageCtx.textBaseline = "top"
                pageCtx.fillText(
                    `${modelResult.name}`,
                    modelResult.x1 ?? 0 + 16 + 128 * modelResult.windowI,
                    modelResult.y - page.y + 16
                )
            }

            if (modelResult.windowTop !== undefined) {
                pageCtx.lineWidth = Math.floor(scansWidth * 0.01)
                pageCtx.strokeStyle = "rgba(0, 100, 0, 0.5)"
                pageCtx.strokeRect(scansWidth, modelResult.windowTop - page.y, 50, 1)
            }
            if (modelResult.windowBottom !== undefined) {
                pageCtx.lineWidth = Math.floor(scansWidth * 0.005)
                pageCtx.strokeStyle = "rgba(0, 100, 0, 0.5)"
                pageCtx.strokeRect(scansWidth + 50, modelResult.windowBottom - page.y, 50, 1)
            }
        }

        for (const modelResult of modelResultsUnmerged) {
            pageCtx.lineWidth = Math.floor(scansWidth * 0.01)
            if (modelResult.panelType === "complex") {
                pageCtx.strokeStyle = `rgba(255, 0, 0, 0.5)`
            } else if (modelResult.panelType === "text") {
                pageCtx.strokeStyle = `rgba(0, 0, 255, 0.5)`
            } else if (modelResult.panelType === "speech_up") {
                pageCtx.strokeStyle = `rgba(255, 0, 255, 0.5)`
            } else if (modelResult.panelType === "speech_down") {
                pageCtx.strokeStyle = `rgba(150, 0, 255, 0.5)`
            }
            pageCtx.strokeRect(
                modelResult.x1 ?? 0,
                modelResult.y - page.y,
                (modelResult.x2 ?? scansWidth) - (modelResult.x1 ?? 0),
                modelResult.height
            )
            if (modelResult.windowI !== undefined && modelResult.name !== undefined) {
                pageCtx.font = "32px serif"
                pageCtx.textBaseline = "top"
                pageCtx.fillText(
                    `${modelResult.name}`,
                    modelResult.x1 ?? 0 + 16,
                    modelResult.y - page.y + 16 + 18 * modelResult.windowI
                )
            }
        }

        if (page.panels[0].pageNum !== undefined) {
            pageCtx.font = "32px serif"
            pageCtx.fillStyle = "blue"
            pageCtx.textBaseline = "bottom"
            pageCtx.fillText(`page ${page.panels[0].pageNum} ${page.panI ?? ""}`, 40, 40)
        }
    }

    if (page.fadeTop) {
        const gradient = pageCtx.createLinearGradient(0, 0, 0, page.height * 0.1)
        gradient.addColorStop(0, "rgba(0, 0, 0, 1)")
        gradient.addColorStop(1, "rgba(0, 0, 0, 0)")
        pageCtx.fillStyle = gradient
        pageCtx.fillRect(0, 0, scansWidth, page.height * 0.1)
    }

    if (page.fadeBottom) {
        const gradient = pageCtx.createLinearGradient(0, page.height * 0.9, 0, page.height)
        gradient.addColorStop(0, "rgba(0, 0, 0, 0)")
        gradient.addColorStop(1, "rgba(0, 0, 0, 1)")
        pageCtx.fillStyle = gradient
        pageCtx.fillRect(0, page.height * 0.9, scansWidth, page.height * 0.1)
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

    return pageCanvas
}

/**
 * @param {ScanLoader[]} scans
 * @param {number} ratio
 * @returns {AsyncGenerator<ScanLoader>}
 */
export async function* renderPages(scans, ratio, stretch) {
    let twoPagesAgo = null
    let onePageAgo = null
    let pageI = 0
    const urlCount = {}
    const scansGenerator = new LoadScansForPanel(scans)
    for await (const page of getPages(scansGenerator, ratio, stretch)) {
        console.log("loaded page", page, pageI)
        if (onePageAgo !== null) {
            yield renderPage(scansGenerator, onePageAgo, twoPagesAgo, page, urlCount)
        }
        twoPagesAgo = onePageAgo
        onePageAgo = page
        pageI++
    }
    if (onePageAgo !== null) {
        yield renderPage(scansGenerator, onePageAgo, twoPagesAgo, null, urlCount)
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
    const startTime = performance.now()

    const width = Math.floor(img.naturalWidth * 0.95)
    const height = img.naturalHeight

    // Create canvas and draw original image
    console.log("a", width, height, img, img.naturalWidth);
    const originalCanvas = new OffscreenCanvas(width, height)
    console.log("B")
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

    const colorTolerance = 10
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
            const y = Math.floor(whitespace[0] + whitespace[1] * 0.5)
            const i = y * 4
            const r = colData[i]
            const g = colData[i + 1]
            const b = colData[i + 2]
            if (
                (Math.abs(r - g) > 10 || Math.abs(r - b) > 10 || Math.abs(g - b) > 10) &&
                !((r < 30 && g < 30 && b < 30) || (r > 225 && g > 225 && b > 225))
            ) {
                return false
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

    const endTime = performance.now()
    console.log(`time findPanels ${endTime - startTime}`)

    return mergedPanels
}
