"use client"

import type React from "react"

import { useState, useEffect, useLayoutEffect, useRef } from "react"
import { gsap } from "gsap"
import Link from "next/link"
import { Mic, Square, Upload, FileCheck2, AlertCircle, Check, Watch, AudioLines, X, ArrowRight } from "lucide-react"
import Results from "./results"
import { analyzeWithProgress, ApiError, pacedProgress, type AnalysisProgress } from "./analysis"
import { CompactProgress, useProgressView } from "./AnalysisProgress"

const DASS21_QUESTIONS = [
  "I found it hard to wind down",                  // q1(S)
  "I tended to over-react to situations",          // q6(s)
  "I felt that I was using a lot of nervous energy", // q8(s)
  "I found myself getting agitated",               // q11(s)
  "I found it difficult to relax",                 // q12(s)
  "I was intolerant of anything that kept me from getting on with what I was doing", // q14(s)
  "I felt that I was rather touchy"                // q18(s)
]

const SCALE = [
  { value: 0, label: "Never" },
  { value: 1, label: "Sometimes" },
  { value: 2, label: "Often" },
  { value: 3, label: "Almost always" },
]

const PHYSIO_EXTENSIONS = [".csv"]
// Deliberate extra wait before results appear (the models themselves take well under a second).
// It is spread over the six progress steps so each real step stays visible a little longer.
const EXTRA_WAIT_MS = 5000
const PROGRESS_STEPS = 6
// Set NEXT_PUBLIC_API_URL (e.g. in Client/.env.local) when the backend is not on localhost:8000.
const API_URL = `${(process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "")}/predict/stream`

// Browsers support different recording formats (Safari records MP4/AAC, not WebM).
// Each maps to a file extension the backend accepts.
const RECORDING_FORMATS = [
  { mimeType: "audio/webm;codecs=opus", extension: ".webm" },
  { mimeType: "audio/webm", extension: ".webm" },
  { mimeType: "audio/mp4", extension: ".m4a" },
  { mimeType: "audio/ogg;codecs=opus", extension: ".ogg" },
]

function pickRecordingFormat() {
  return RECORDING_FORMATS.find((format) => MediaRecorder.isTypeSupported(format.mimeType))
}

function extensionForMimeType(mimeType: string) {
  if (mimeType.includes("mp4") || mimeType.includes("aac")) return ".m4a"
  if (mimeType.includes("ogg")) return ".ogg"
  return ".webm"
}

function microphoneErrorMessage(err: unknown) {
  const name = err instanceof DOMException ? err.name : ""
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Microphone access was blocked. Allow it in your browser's site settings, or upload an audio file instead."
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "No microphone was found. Connect one, or upload an audio file instead."
  }
  return "Recording isn't available in this browser. Upload an audio file instead."
}

export default function CheckPage() {
  const [deviceConnected, setDeviceConnected] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  // null = not answered yet; every statement must be answered (0 "Never" is a valid answer).
  const [dass21Responses, setDass21Responses] = useState<(number | null)[]>(new Array(DASS21_QUESTIONS.length).fill(null))
  const [stressResult, setStressResult] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [audioSource, setAudioSource] = useState<"recording" | "upload" | null>(null)
  const [analyzedAt, setAnalyzedAt] = useState<Date | null>(null)
  const [progress, setProgress] = useState<AnalysisProgress | null>(null)
  const progressView = useProgressView(progress)
  const analysisAbortRef = useRef<AbortController | null>(null)
  const pacingRef = useRef<ReturnType<typeof pacedProgress> | null>(null)
  const [audioURL, setAudioURL] = useState<string | null>(null)
  const [recordingTime, setRecordingTime] = useState(0)
  const [micError, setMicError] = useState<string | null>(null)
  const [allDataReady, setAllDataReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dropError, setDropError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  // Audio recording refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioURLRef = useRef<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const timerRef = useRef<NodeJS.Timeout | null>(null)

  const pageRef = useRef<HTMLDivElement>(null)
  const resultsRef = useRef<HTMLDivElement>(null)

  useLayoutEffect(() => {
    const mm = gsap.matchMedia()
    mm.add("(prefers-reduced-motion: no-preference)", () => {
      gsap.from(".check-section", { y: 32, opacity: 0, duration: 0.9, stagger: 0.1, ease: "expo.out" })
    }, pageRef)
    return () => mm.revert()
  }, [])

  useEffect(() => {
    if (stressResult && resultsRef.current) {
      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches
      resultsRef.current.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" })
      if (!reduceMotion) {
        gsap.fromTo(resultsRef.current, { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.8, ease: "expo.out" })
      }
    }
  }, [stressResult])

  // All three inputs are required before analysis can start.
  useEffect(() => {
    const hasPhysiological = uploadedFile !== null
    const hasQuestionnaire = dass21Responses.every((response) => response !== null)
    const hasVoice = audioFile !== null

    setAllDataReady(hasPhysiological && hasQuestionnaire && hasVoice)
  }, [uploadedFile, dass21Responses, audioFile])

  const handleDass21Change = (index: number, value: number) => {
    const newResponses = [...dass21Responses]
    newResponses[index] = value
    setDass21Responses(newResponses)
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    // Reset so choosing the same file again still fires onChange.
    event.target.value = ""
    if (file) {
      setDropError(null)
      setUploadedFile(file)
    }
  }

  const handleFileDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setIsDragging(false)
    const file = event.dataTransfer.files?.[0]
    if (!file) return
    if (!PHYSIO_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext))) {
      setDropError("That file type isn't supported. Please use a .csv file.")
      return
    }
    setDropError(null)
    setUploadedFile(file)
  }

  // Keeps the playback URL in sync with the current audio and frees the previous one.
  const replaceAudio = (file: File | null, source: "recording" | "upload" | null) => {
    if (audioURLRef.current) URL.revokeObjectURL(audioURLRef.current)
    const url = file ? URL.createObjectURL(file) : null
    audioURLRef.current = url
    setAudioURL(url)
    setAudioFile(file)
    setAudioSource(source)
  }

  const handleAudioUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (file) {
      setMicError(null)
      replaceAudio(file, "upload")
    }
  }

  const simulateDeviceConnection = () => {
    setDeviceConnected(!deviceConnected)
  }

  const releaseMicrophone = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
  }

  const startRecording = async () => {
    setMicError(null)
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      // getUserMedia is only available on HTTPS or localhost.
      setMicError("Recording needs a secure (HTTPS) connection in a supported browser. Upload an audio file instead.")
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100
        }
      });
      streamRef.current = stream

      const format = pickRecordingFormat()
      const recorder = new MediaRecorder(stream, format ? { mimeType: format.mimeType } : undefined)
      mediaRecorderRef.current = recorder

      const chunks: Blob[] = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      recorder.onstop = () => {
        releaseMicrophone()
        const mimeType = recorder.mimeType || format?.mimeType || "audio/webm"
        const type = mimeType.split(";")[0]
        const blob = new Blob(chunks, { type })
        if (blob.size === 0) {
          setMicError("Nothing was recorded. Please try again.")
          return
        }
        replaceAudio(new File([blob], `recorded_audio${extensionForMimeType(mimeType)}`, { type }), "recording")
      };

      recorder.start();
      setIsRecording(true);
      setRecordingTime(0);

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (err) {
      console.error('Error starting recording:', err);
      releaseMicrophone()
      setMicError(microphoneErrorMessage(err));
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  // Leaving the page mid-recording must turn the microphone off and free the playback URL.
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      const recorder = mediaRecorderRef.current
      if (recorder && recorder.state !== "inactive") {
        recorder.onstop = null
        recorder.stop()
      }
      streamRef.current?.getTracks().forEach((track) => track.stop())
      if (audioURLRef.current) URL.revokeObjectURL(audioURLRef.current)
      analysisAbortRef.current?.abort()
      pacingRef.current?.cancel()
    }
  }, [])

  const analyzeAllModalities = async () => {
    if (!allDataReady || !uploadedFile || !audioFile) return;

    setIsLoading(true);
    setError(null);
    setStressResult(null);
    try {
      // Field names and formats must match Server/main.py POST /predict.
      const formData = new FormData();
      formData.append("physiological_file", uploadedFile);
      formData.append("dass21_responses", dass21Responses.map((value) => value ?? 0).join(","));
      formData.append("voice_audio", audioFile, audioFile.name);

      const controller = new AbortController()
      analysisAbortRef.current = controller
      // Validation failures carry a readable "message" explaining which input to fix (see ApiError).
      const pacing = pacedProgress(setProgress, EXTRA_WAIT_MS / PROGRESS_STEPS)
      pacingRef.current = pacing
      const result = await analyzeWithProgress(API_URL, formData, pacing.report, controller.signal)
      await pacing.finished
      setStressResult(result);
      setAnalyzedAt(new Date());
    } catch (error) {
      // Errors are shown straight away rather than after the pacing delay.
      pacingRef.current?.cancel()
      if (error instanceof DOMException && error.name === "AbortError") return
      console.error("Stress analysis failed:", error);
      setError(
        error instanceof ApiError
          ? error.detail ?? `The analysis service returned an error (status ${error.status}). Please check your inputs and try again.`
          : "Something went wrong while analyzing. Make sure the SafeSpace API is running, then try again.",
      );
    } finally {
      setIsLoading(false);
    }
  };

  const getCompletionPercentage = () => {
    const done = [uploadedFile !== null, dass21Responses.every((response) => response !== null), audioFile !== null]
    return Math.round((done.filter(Boolean).length / done.length) * 100)
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const clearAudio = () => {
    replaceAudio(null, null)
    setRecordingTime(0);
  };

  const answeredCount = dass21Responses.filter((response) => response !== null).length
  const hasQuestionnaire = answeredCount === DASS21_QUESTIONS.length
  const completion = getCompletionPercentage()

  const checklist = [
    { id: "step-physio", label: "Physiological data", done: !!uploadedFile, required: true },
    { id: "step-questionnaire", label: "Questionnaire", done: hasQuestionnaire, required: true },
    { id: "step-voice", label: "Voice", done: !!audioFile, required: true },
  ]

  const statusMessage = isLoading
    ? `Analyzing: ${progressView?.label ?? "starting"}.`
    : error
      ? error
      : stressResult?.predictions?.prediction_label
        ? `Analysis complete. Predicted stress level: ${stressResult.predictions.prediction_label}.`
        : ""

  return (
    <main ref={pageRef} id="main" className="min-h-screen pb-24 pt-28 sm:pt-32">
      <div className="mx-auto max-w-7xl px-5 sm:px-8 lg:px-10">
        {/* Header */}
        <header className="check-section grid gap-6 border-b border-ink/15 pb-10 lg:grid-cols-12 lg:pb-14">
          <p className="eyebrow lg:col-span-4 lg:pt-4">Stress check · 3 steps</p>
          <div className="lg:col-span-8">
            <h1 className="font-display text-[clamp(2.5rem,6vw,4.75rem)] leading-[0.98] tracking-[-0.03em] text-ink">
              Let&rsquo;s see how you&rsquo;re <span className="italic text-pine">really</span> doing.
            </h1>
            <p className="mt-6 max-w-2xl text-lg leading-relaxed text-ink/70">
              Add your physiological data, answer seven short statements, and include a short voice sample to get your reading.
            </p>
            <Link href="/stress-buster" className="link-draw mt-5 inline-flex items-center gap-2 text-sm font-semibold text-ink">
              Need a break first? Try StressBuster
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Link>
          </div>
        </header>

        <div className="mt-10 grid gap-10 lg:mt-14 lg:grid-cols-12 lg:gap-12">
          {/* Steps */}
          <div className="min-w-0 space-y-6 lg:col-span-8">
            {/* 01 Physiological */}
            <StepSection
              id="step-physio"
              number="01"
              title="Physiological data"
              done={!!uploadedFile}
              required
              intro="Wearable signals such as EDA, ECG, temperature and movement, exported as a CSV file."
            >
              <div
                onDragOver={(e) => {
                  e.preventDefault()
                  setIsDragging(true)
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleFileDrop}
              >
                <input
                  id="file-upload"
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="peer sr-only"
                  aria-describedby="file-upload-hint"
                />
                <label
                  htmlFor="file-upload"
                  className={`flex cursor-pointer flex-col items-start gap-4 rounded-2xl border-[1.5px] border-dashed p-6 transition-colors duration-300 peer-focus-visible:ring-2 peer-focus-visible:ring-ring peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-card sm:flex-row sm:items-center sm:p-8 ${
                    isDragging
                      ? "border-pine bg-sage-soft"
                      : uploadedFile
                        ? "border-pine/40 bg-sage-soft/60"
                        : "border-ink/20 hover:border-ink/40 hover:bg-muted/50"
                  }`}
                >
                  <span
                    className={`grid h-12 w-12 shrink-0 place-items-center rounded-full ${uploadedFile ? "bg-pine text-paper" : "bg-muted text-ink"}`}
                  >
                    {uploadedFile ? <FileCheck2 className="h-5 w-5" aria-hidden="true" /> : <Upload className="h-5 w-5" aria-hidden="true" />}
                  </span>
                  <span className="min-w-0 flex-1">
                    {uploadedFile ? (
                      <>
                        <span className="block truncate font-semibold text-ink">{uploadedFile.name}</span>
                        <span className="mt-0.5 block text-sm text-muted-foreground">
                          {formatBytes(uploadedFile.size)} · uploaded
                        </span>
                      </>
                    ) : (
                      <>
                        <span className="block font-semibold text-ink">Upload physiological data</span>
                        <span id="file-upload-hint" className="mt-0.5 block text-sm text-muted-foreground">
                          Drop a CSV file (ECG, EDA, EMG, Temp at 100 Hz) here, or browse.
                        </span>
                      </>
                    )}
                  </span>
                  <span className="rounded-full border border-ink/20 px-4 py-2 text-sm font-semibold text-ink">
                    {uploadedFile ? "Replace" : "Browse"}
                  </span>
                </label>
                {dropError && (
                  <p role="alert" className="mt-3 flex items-center gap-2 text-sm text-clay-deep">
                    <AlertCircle className="h-4 w-4 shrink-0" aria-hidden="true" />
                    {dropError}
                  </p>
                )}
              </div>

              <div className="mt-5 flex items-center justify-between gap-4 rounded-2xl bg-muted/60 px-5 py-4">
                <div className="flex items-center gap-3">
                  <Watch className={`h-5 w-5 ${deviceConnected ? "text-pine" : "text-ink/40"}`} aria-hidden="true" />
                  <div>
                    <p id="device-label" className="text-sm font-semibold text-ink">
                      Wearable device
                    </p>
                    <p className="text-sm text-muted-foreground">{deviceConnected ? "Connected" : "Not connected"}</p>
                  </div>
                </div>
                <button
                  type="button"
                  role="switch"
                  aria-checked={deviceConnected}
                  aria-labelledby="device-label"
                  onClick={simulateDeviceConnection}
                  className={`relative h-7 w-12 shrink-0 rounded-full transition-colors duration-300 ${deviceConnected ? "bg-pine" : "bg-ink/20"}`}
                >
                  <span
                    className={`absolute left-1 top-1 h-5 w-5 rounded-full bg-paper shadow transition-transform duration-300 ease-soft ${deviceConnected ? "translate-x-5" : ""}`}
                  />
                </button>
              </div>
            </StepSection>

            {/* 02 Questionnaire */}
            <StepSection
              id="step-questionnaire"
              number="02"
              title="Questionnaire"
              done={hasQuestionnaire}
              required
              intro="Seven statements from the DASS-21 stress scale. How often did each apply to you recently?"
            >
              <ol className="divide-y divide-ink/10 border-y border-ink/10">
                {DASS21_QUESTIONS.map((question, index) => (
                  <li key={index} className="py-6">
                    <fieldset>
                      <legend className="float-left mb-4 flex w-full gap-3 text-base font-medium leading-snug text-ink sm:text-lg">
                        <span className="pt-0.5 font-mono text-xs text-muted-foreground sm:pt-1">{String(index + 1).padStart(2, "0")}</span>
                        <span>{question}</span>
                      </legend>
                      <div className="clear-both grid grid-cols-4 gap-1.5 sm:gap-2">
                        {SCALE.map((option) => {
                          const inputId = `q${index}-${option.value}`
                          const checked = dass21Responses[index] === option.value
                          return (
                            <div key={option.value}>
                              <input
                                type="radio"
                                id={inputId}
                                name={`dass21-${index}`}
                                value={option.value}
                                checked={checked}
                                onChange={() => handleDass21Change(index, option.value)}
                                className="peer sr-only"
                              />
                              <label
                                htmlFor={inputId}
                                className={`flex h-full min-h-[64px] cursor-pointer flex-col items-center justify-center rounded-xl border px-1 py-2 text-center transition-[background-color,border-color,color,transform] duration-200 peer-focus-visible:ring-2 peer-focus-visible:ring-ring peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-card motion-safe:active:scale-[0.97] ${
                                  checked
                                    ? "border-pine bg-pine text-paper"
                                    : "border-ink/15 text-ink hover:border-ink/40 hover:bg-muted/60"
                                }`}
                              >
                                <span className="font-display text-xl leading-none sm:text-2xl">{option.value}</span>
                                <span className={`mt-1 text-[0.7rem] leading-tight sm:text-xs ${checked ? "text-paper/85" : "text-muted-foreground"}`}>
                                  {option.label}
                                </span>
                              </label>
                            </div>
                          )
                        })}
                      </div>
                    </fieldset>
                  </li>
                ))}
              </ol>
            </StepSection>

            {/* 03 Voice */}
            <StepSection
              id="step-voice"
              number="03"
              title="Voice"
              done={!!audioFile}
              required
              intro="Record a few sentences about your day or upload a short voice sample. You need one to continue."
            >
              {!audioURL ? (
                <div className="flex flex-col gap-6 sm:flex-row sm:items-center">
                  <button
                    type="button"
                    onClick={toggleRecording}
                    aria-pressed={isRecording}
                    className={`group relative grid h-24 w-24 shrink-0 place-items-center rounded-full text-paper transition-[background-color,transform] duration-300 ease-soft motion-safe:hover:scale-[1.04] ${
                      isRecording ? "bg-clay-deep" : "bg-pine hover:bg-pine-deep"
                    }`}
                  >
                    {isRecording && <span className="absolute inset-0 rounded-full bg-clay/40 motion-safe:animate-ping" aria-hidden="true" />}
                    {isRecording ? <Square className="relative h-6 w-6 fill-current" aria-hidden="true" /> : <Mic className="relative h-7 w-7" aria-hidden="true" />}
                    <span className="sr-only">{isRecording ? "Stop recording" : "Start voice recording"}</span>
                  </button>

                  <div className="min-w-0 flex-1">
                    <div className="flex h-8 items-end gap-1" aria-hidden="true">
                      {Array.from({ length: 18 }, (_, i) => (
                        <span
                          key={i}
                          className={`w-1.5 rounded-full transition-colors duration-300 ${isRecording ? "animate-level bg-clay" : "bg-ink/15"}`}
                          style={{
                            height: `${30 + ((i * 37) % 70)}%`,
                            animationDelay: `${(i % 6) * -0.15}s`,
                          }}
                        />
                      ))}
                    </div>
                    <p className="mt-3 flex items-baseline gap-3 text-sm font-medium text-ink" aria-live="polite">
                      {isRecording ? "Recording… tap to stop." : "Tap the microphone to record."}
                      {isRecording && <span className="font-mono tabular-nums text-clay-deep">{formatTime(recordingTime)}</span>}
                    </p>
                    {micError && (
                      <p role="alert" className="mt-2 flex items-start gap-2 text-sm text-clay-deep">
                        <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                        {micError}
                      </p>
                    )}
                  </div>
                </div>
              ) : (
                <div className="min-w-0 rounded-2xl bg-sage-soft/70 p-4 sm:p-5">
                  <div className="flex items-center gap-3">
                    <audio ref={audioRef} src={audioURL} controls className="h-10 w-full min-w-0 flex-1" />
                    <button
                      type="button"
                      onClick={clearAudio}
                      aria-label="Remove recording"
                      className="grid h-10 w-10 shrink-0 place-items-center rounded-full text-ink/60 transition-colors hover:bg-ink/5 hover:text-clay-deep"
                    >
                      <X className="h-4 w-4" aria-hidden="true" />
                    </button>
                  </div>
                  <p className="mt-3 flex items-center gap-2 text-sm text-ink" aria-live="polite">
                    <Check className="h-4 w-4 text-pine" strokeWidth={3} aria-hidden="true" />
                    {audioSource === "recording"
                      ? "Voice recorded successfully. Play it back to check before submitting."
                      : `${audioFile?.name ?? "Audio"} uploaded. Play it back to check before submitting.`}
                  </p>
                </div>
              )}

              <div className="mt-6 flex flex-col gap-3 border-t border-ink/10 pt-6 sm:flex-row sm:items-center sm:justify-between">
                <p className="text-sm text-muted-foreground">Prefer a file? WAV, MP3, M4A, FLAC, OGG or WebM. The first 5 seconds are analysed.</p>
                <div>
                  <input
                    id="audio-upload"
                    type="file"
                    accept=".wav,.mp3,.m4a,.flac,.ogg,.webm"
                    onChange={handleAudioUpload}
                    disabled={isRecording}
                    className="peer sr-only"
                  />
                  <label
                    htmlFor="audio-upload"
                    aria-disabled={isRecording}
                    className="btn-ghost cursor-pointer aria-disabled:cursor-not-allowed aria-disabled:opacity-50 !min-h-[40px] !py-2 text-sm peer-focus-visible:ring-2 peer-focus-visible:ring-ring peer-focus-visible:ring-offset-2 peer-focus-visible:ring-offset-card"
                  >
                    <AudioLines className="h-4 w-4" aria-hidden="true" />
                    Upload audio file
                  </label>
                </div>
              </div>
            </StepSection>
          </div>

          {/* Summary */}
          <aside className="check-section lg:col-span-4" aria-label="Check summary">
            <div className="rounded-3xl bg-ink p-6 text-paper sm:p-8 lg:sticky lg:top-24">
              <div className="flex items-end justify-between">
                <p className="eyebrow !text-paper/60">Your check</p>
                <p className="font-display text-4xl leading-none tracking-tight">
                  {completion}
                  <span className="text-xl text-paper/60">%</span>
                </p>
              </div>
              <div
                className="mt-4 h-1.5 overflow-hidden rounded-full bg-paper/15"
                role="progressbar"
                aria-label="Check completion"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={completion}
              >
                <div className="h-full rounded-full bg-sage transition-[width] duration-700 ease-soft" style={{ width: `${completion}%` }} />
              </div>

              <ul className="mt-6 divide-y divide-paper/10 border-y border-paper/10">
                {checklist.map((item) => (
                  <li key={item.id}>
                    <a href={`#${item.id}`} className="flex items-center justify-between gap-3 py-3.5 transition-colors hover:text-sage">
                      <span className="flex items-center gap-3">
                        <span
                          className={`grid h-5 w-5 place-items-center rounded-full border transition-colors duration-300 ${
                            item.done ? "border-sage bg-sage text-ink" : "border-paper/30"
                          }`}
                          aria-hidden="true"
                        >
                          {item.done && <Check className="h-3 w-3" strokeWidth={3} />}
                        </span>
                        <span className="text-[0.95rem]">{item.label}</span>
                        <span className="sr-only">{item.done ? "(done)" : "(not done)"}</span>
                      </span>
                      <span className="font-mono text-[0.68rem] uppercase tracking-[0.12em] text-paper/50">
                        {item.required ? "Required" : "Optional"}
                      </span>
                    </a>
                  </li>
                ))}
              </ul>

              {error && (
                <div className="mt-6 flex gap-3 rounded-2xl bg-clay-soft p-4 text-sm text-clay-deep">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                  <p>{error}</p>
                </div>
              )}

              <button
                type="button"
                onClick={analyzeAllModalities}
                disabled={!allDataReady || isLoading}
                className="btn mt-6 w-full bg-sage text-base text-ink hover:bg-sage-soft focus-visible:outline-paper disabled:bg-paper/15 disabled:text-paper/60 disabled:opacity-100"
              >
                {isLoading ? (
                  <>
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-ink/30 border-t-ink" aria-hidden="true" />
                    Analyzing…
                  </>
                ) : (
                  "Analyze stress level"
                )}
              </button>

              {isLoading && progressView && <CompactProgress view={progressView} />}

              {!allDataReady && (
                <p className="mt-4 text-sm leading-relaxed text-paper/60">
                  {uploadedFile && !hasQuestionnaire
                    ? `Answer all seven statements to continue (${answeredCount} of 7 answered).`
                    : uploadedFile && hasQuestionnaire && !audioFile
                      ? "Record or upload a voice sample to continue."
                      : "Add physiological data, answer the questionnaire, and include a voice sample to continue."}
                </p>
              )}

              <p className="sr-only" role="status" aria-live="polite">
                {statusMessage}
              </p>
            </div>
          </aside>
        </div>

        {/* Results */}
        <section ref={resultsRef} id="results" aria-labelledby="results-heading" className="mt-20 scroll-mt-24 lg:mt-28">
          <div className="mb-8 flex items-baseline justify-between gap-4 border-b border-ink/15 pb-6">
            <h2 id="results-heading" className="font-display text-3xl tracking-[-0.02em] text-ink sm:text-4xl">
              Your reading
            </h2>
            <p className="eyebrow">Results</p>
          </div>
          <Results result={stressResult} isLoading={isLoading} analyzedAt={analyzedAt} progress={progressView} />
        </section>
      </div>
    </main>
  )
}

function StepSection({
  id,
  number,
  title,
  intro,
  done,
  required = false,
  children,
}: {
  id: string
  number: string
  title: string
  intro: string
  done: boolean
  required?: boolean
  children: React.ReactNode
}) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="check-section scroll-mt-24 rounded-3xl border border-ink/10 bg-card p-5 sm:p-8">
      <div className="mb-6 flex items-start justify-between gap-4">
        <div className="flex gap-4 sm:gap-5">
          <span className="font-display text-2xl leading-none text-ink/30 sm:text-3xl">{number}</span>
          <div>
            <h2 id={`${id}-title`} className="font-display text-2xl leading-tight tracking-[-0.01em] text-ink sm:text-3xl">
              {title}
            </h2>
            <p className="mt-2 max-w-xl text-[0.95rem] leading-relaxed text-muted-foreground">{intro}</p>
          </div>
        </div>
        <span
          className={`inline-flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1 font-mono text-[0.68rem] uppercase tracking-[0.12em] transition-colors duration-300 ${
            done ? "bg-pine text-paper" : "border border-ink/15 text-ink/60"
          }`}
        >
          {done && <Check className="h-3 w-3" strokeWidth={3} aria-hidden="true" />}
          {done ? "Done" : required ? "Required" : "Optional"}
        </span>
      </div>
      {children}
    </section>
  )
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
