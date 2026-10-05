// Client for POST /predict/stream (Server/main.py). XMLHttpRequest is used instead of fetch
// because it reports upload progress, and its responseText grows as the server streams events.

export type AnalysisStage =
  | "uploading"
  | "queued"
  | "physiological"
  | "questionnaire"
  | "voice"
  | "fusion"
  | "explanations"
  | "done"

export interface AnalysisProgress {
  stage: AnalysisStage
  /** 0-1, only for the "uploading" stage */
  uploadFraction?: number
}

export class ApiError extends Error {
  constructor(public status: number, public detail?: string) {
    super(`HTTP ${status}`)
  }
}

export class NetworkError extends Error {}

const SERVER_STAGES: AnalysisStage[] = ["physiological", "questionnaire", "voice", "fusion", "explanations"]

interface StreamEvent {
  type: "queued" | "stage" | "result"
  stage?: string
  status?: number
  body?: any
}

export function analyzeWithProgress(
  url: string,
  formData: FormData,
  onProgress: (progress: AnalysisProgress) => void,
  signal?: AbortSignal,
): Promise<any> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    let consumed = 0
    let result: StreamEvent | null = null
    let malformed = false

    const readEvents = () => {
      if (xhr.status !== 200) return
      const text = xhr.responseText
      let newline: number
      while ((newline = text.indexOf("\n", consumed)) !== -1) {
        const line = text.slice(consumed, newline).trim()
        consumed = newline + 1
        if (!line) continue
        let event: StreamEvent
        try {
          event = JSON.parse(line)
        } catch {
          malformed = true
          continue
        }
        if (event.type === "queued") onProgress({ stage: "queued" })
        else if (event.type === "stage" && SERVER_STAGES.includes(event.stage as AnalysisStage)) {
          onProgress({ stage: event.stage as AnalysisStage })
        } else if (event.type === "result") result = event
      }
    }

    xhr.open("POST", url)
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress({ stage: "uploading", uploadFraction: event.loaded / event.total })
    }
    xhr.onprogress = readEvents
    xhr.onload = () => {
      if (xhr.status !== 200) {
        // Upload errors (413, 422) come back as a normal JSON body before any streaming.
        let body: any = null
        try {
          body = JSON.parse(xhr.responseText)
        } catch {}
        reject(new ApiError(xhr.status, body?.message))
        return
      }
      readEvents()
      if (!result || malformed) {
        reject(new ApiError(502, "The analysis ended unexpectedly. Please try again."))
      } else if (result.status !== 200) {
        reject(new ApiError(result.status ?? 500, result.body?.message))
      } else {
        onProgress({ stage: "done" })
        resolve(result.body)
      }
    }
    xhr.onerror = () => reject(new NetworkError("Network error"))
    xhr.onabort = () => reject(new DOMException("Analysis cancelled", "AbortError"))
    signal?.addEventListener("abort", () => xhr.abort(), { once: true })

    onProgress({ stage: "uploading", uploadFraction: 0 })
    xhr.send(formData)
  })
}

/**
 * Delays progress events so each step stays on screen `stepDelayMs` longer than it really took.
 * Step k (0 = upload, 1-5 = server stages) is shown at its arrival time + k * stepDelayMs, and
 * "done" at its arrival time + 6 * stepDelayMs, so a result appears 6 * stepDelayMs later than
 * it otherwise would. Events keep their order; nothing is shown before the server reports it.
 */
export function pacedProgress(onProgress: (progress: AnalysisProgress) => void, stepDelayMs: number) {
  const timers: ReturnType<typeof setTimeout>[] = []
  let stagesSeen = 0
  let lastShownAt = 0
  let resolveFinished: () => void = () => {}
  const finished = new Promise<void>((resolve) => {
    resolveFinished = resolve
  })

  const report = (progress: AnalysisProgress) => {
    if (SERVER_STAGES.includes(progress.stage)) stagesSeen += 1
    const slots = progress.stage === "done" ? SERVER_STAGES.length + 1 : stagesSeen
    const showAt = Math.max(performance.now() + slots * stepDelayMs, lastShownAt)
    lastShownAt = showAt
    timers.push(
      setTimeout(() => {
        onProgress(progress)
        if (progress.stage === "done") resolveFinished()
      }, showAt - performance.now()),
    )
  }

  const cancel = () => timers.forEach(clearTimeout)

  return { report, finished, cancel }
}
