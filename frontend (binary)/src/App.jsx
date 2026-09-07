import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import JSZip from "jszip";
import {
  Upload,
  Image as ImageIcon,
  Sparkles,
  WandSparkles,
  SlidersHorizontal,
  Download,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Info,
  ChevronDown,
  Zap,
  ScanSearch,
  Settings2,
  Trash2,
  Circle,
  CheckCircle,
  X,
  Images,
  Link,
  Globe,
} from "lucide-react";

let API = null;

const FULL_POWER = {
  film_strength: 1.0, highlight_rolloff: 1.0, shadow_tint: 1.0, color_separation: 1.0,
  halation: 1.0, bloom: 1.0, grain: 0.8, vignette: 1.0,
  lens_distortion: 0.5, chromatic_aberration: 0.5, edge_softness: 0.5, seed: 42,
};

const CUSTOM_DEFAULTS = {
  film_strength: 0.75, highlight_rolloff: 0.65, shadow_tint: 0.35, color_separation: 0.28,
  halation: 0.18, bloom: 0.13, grain: 0.32, vignette: 0.17,
  lens_distortion: 0.04, chromatic_aberration: 0.03, edge_softness: 0.1, seed: 42,
};

const PARAMETER_INFO = {
  film_strength: { label: "Film Strength", description: "Controls the intensity of the cinematic film-response curve.", group: "Film" },
  highlight_rolloff: { label: "Highlight Rolloff", description: "Softens bright highlights for a smoother film-like response.", group: "Film" },
  shadow_tint: { label: "Shadow Tint", description: "Adds subtle cinematic color bias to darker regions.", group: "Film" },
  color_separation: { label: "Color Separation", description: "Controls subtle per-channel color differentiation.", group: "Film" },
  halation: { label: "Halation", description: "Creates a warm film-like glow around strong highlights.", group: "Optical" },
  bloom: { label: "Bloom", description: "Adds soft light diffusion around bright image regions.", group: "Optical" },
  grain: { label: "Film Grain", description: "Adds luminance-oriented film grain.", group: "Optical" },
  vignette: { label: "Vignette", description: "Darkens the outer frame gradually to emphasize the center.", group: "Optical" },
  lens_distortion: { label: "Lens Distortion", description: "Simulates subtle barrel/pincushion lens characteristics.", group: "Geometry" },
  chromatic_aberration: { label: "Chromatic Aberration", description: "Adds subtle color fringing near image edges.", group: "Geometry" },
  edge_softness: { label: "Edge Softness", description: "Softens peripheral detail for a gentler optical rendering.", group: "Geometry" },
};

const PIPELINE_STAGES = [
  { key: "analyzing", label: "Scene Analysis" },
  { key: "planning", label: "Render Plan" },
  { key: "lut", label: "Adaptive 3D LUT" },
  { key: "film", label: "Film Transform" },
  { key: "optical", label: "Optical Effects" },
  { key: "geometry", label: "Geometry" },
  { key: "saving", label: "Saving" },
];

function Tooltip({ text }) {
  return (
    <span className="group relative inline-flex">
      <Info className="h-3.5 w-3.5 text-violet-300/70" />
      <span className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-2 hidden w-64 -translate-x-1/2 rounded-xl border border-violet-400/20 bg-[#140b21] p-3 text-xs leading-relaxed text-violet-50 shadow-2xl group-hover:block">{text}</span>
    </span>
  );
}

function SliderParameter({ name, value, onChange }) {
  const info = PARAMETER_INFO[name];
  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.025] p-4 transition hover:border-violet-400/20 hover:bg-white/[0.035]">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-white">{info.label}</span>
          <Tooltip text={info.description} />
        </div>
        <span className="rounded-lg bg-violet-500/10 px-2 py-1 font-mono text-xs text-violet-200">{Number(value).toFixed(3)}</span>
      </div>
      <input type="range" min="0" max="1" step="0.001" value={value} onChange={(e) => onChange(Number(e.target.value))} className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-violet-950 accent-violet-500" />
      <div className="mt-2 flex justify-between text-[10px] uppercase tracking-wider text-white/30"><span>0</span><span>Max</span></div>
    </div>
  );
}

function InputSource({ inputMode, setInputMode, files, setFiles, urls, setUrls }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const acceptFiles = useCallback((incoming) => {
    if (!incoming) return;
    const valid = Array.from(incoming).filter((f) => f.type.startsWith("image/"));
    if (valid.length === 0) { alert("Please select image files."); return; }
    setFiles([...files, ...valid]);
  }, [files, setFiles]);

  const addUrls = useCallback((text) => {
    const newUrls = text.split("\n").map((u) => u.trim()).filter(Boolean);
    const existing = urls.split("\n").map((u) => u.trim()).filter(Boolean);
    const combined = [...new Set([...existing, ...newUrls])];
    setUrls(combined.join("\n"));
  }, [urls, setUrls]);

  return (
    <div className="rounded-3xl border border-white/10 bg-white/[0.025] overflow-hidden">
      <div className="flex border-b border-white/8">
        <button onClick={() => setInputMode("files")} className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition ${inputMode === "files" ? "bg-violet-500/10 text-violet-200 border-b-2 border-violet-400" : "text-white/40 hover:text-white/60"}`}>
          <Images className="h-4 w-4" /> Upload Files
        </button>
        <button onClick={() => setInputMode("urls")} className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition ${inputMode === "urls" ? "bg-violet-500/10 text-violet-200 border-b-2 border-violet-400" : "text-white/40 hover:text-white/60"}`}>
          <Globe className="h-4 w-4" /> From URLs
        </button>
      </div>

      {inputMode === "files" ? (
        <div
          onDragEnter={(e) => { e.preventDefault(); setDragging(true); }}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={(e) => { e.preventDefault(); setDragging(false); }}
          onDrop={(e) => { e.preventDefault(); setDragging(false); acceptFiles(e.dataTransfer.files); }}
          className={dragging ? "bg-violet-500/10" : ""}
        >
          <input ref={inputRef} type="file" accept="image/*" multiple className="hidden" onChange={(e) => acceptFiles(e.target.files)} />

          {files.length > 0 ? (
            <div className="p-5">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Images className="h-5 w-5 text-violet-300" />
                  <span className="text-sm font-semibold text-white">{files.length} image{files.length !== 1 ? "s" : ""} selected</span>
                </div>
                <div className="flex gap-2">
                  <button onClick={() => inputRef.current?.click()} className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-1.5 text-xs text-white/60 transition hover:border-violet-400/20 hover:text-white">
                    <Upload className="h-3.5 w-3.5" /> Add more
                  </button>
                  <button onClick={() => setFiles([])} className="inline-flex items-center gap-1.5 rounded-lg border border-red-400/15 bg-red-400/5 px-3 py-1.5 text-xs text-red-300 transition hover:bg-red-400/10">
                    <Trash2 className="h-3.5 w-3.5" /> Clear all
                  </button>
                </div>
              </div>
              <div className="grid max-h-[300px] grid-cols-2 gap-3 overflow-y-auto sm:grid-cols-3 md:grid-cols-4">
                {files.map((f, i) => (
                  <div key={`${f.name}-${i}`} className="group relative overflow-hidden rounded-xl border border-white/8 bg-black/20">
                    <img src={URL.createObjectURL(f)} alt={f.name} className="aspect-square w-full object-cover" />
                    <button onClick={() => setFiles(files.filter((_, j) => j !== i))} className="absolute right-1.5 top-1.5 flex h-6 w-6 items-center justify-center rounded-full bg-black/60 text-white/60 opacity-0 transition hover:bg-red-500 hover:text-white group-hover:opacity-100">
                      <X className="h-3.5 w-3.5" />
                    </button>
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-2 py-1.5">
                      <p className="truncate text-[10px] text-white/70">{f.name}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <button onClick={() => inputRef.current?.click()} className="flex min-h-[220px] w-full flex-col items-center justify-center px-6 text-center">
              <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl border border-violet-400/20 bg-violet-500/10 shadow-xl shadow-violet-900/20">
                <Upload className="h-7 w-7 text-violet-300" />
              </div>
              <h3 className="text-lg font-semibold text-white">Drop your photos here</h3>
              <p className="mt-2 max-w-md text-sm text-white/40">Upload multiple JPG, JPEG, PNG, or WebP images at once.</p>
              <span className="mt-5 inline-flex items-center gap-2 rounded-xl border border-violet-400/20 bg-violet-500/10 px-4 py-2 text-sm text-violet-200">
                <ImageIcon className="h-4 w-4" /> Browse images
              </span>
            </button>
          )}
        </div>
      ) : (
        <div className="p-5">
          <div className="mb-3 flex items-center gap-2">
            <Link className="h-5 w-5 text-violet-300" />
            <span className="text-sm font-semibold text-white">Image URLs</span>
          </div>
          <p className="mb-3 text-xs text-white/40">Paste image URLs, one per line. Engine will download, process, and clean up automatically.</p>
          <textarea
            value={urls}
            onChange={(e) => setUrls(e.target.value)}
            placeholder={"https://example.com/photo1.jpg\nhttps://example.com/photo2.jpg\nhttps://example.com/photo3.jpg"}
            rows={6}
            className="w-full rounded-2xl border border-white/10 bg-[#10091a] px-4 py-3 text-sm text-white placeholder-white/20 outline-none transition focus:border-violet-400/50 resize-none font-mono"
          />
          <div className="mt-3 flex items-center justify-between">
            <span className="text-xs text-white/30">{urls.split("\n").filter((u) => u.trim()).length} URL{urls.split("\n").filter((u) => u.trim()).length !== 1 ? "s" : ""}</span>
            {urls.trim() && (
              <button onClick={() => setUrls("")} className="inline-flex items-center gap-1.5 text-xs text-white/40 hover:text-red-300 transition">
                <Trash2 className="h-3 w-3" /> Clear
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function formatBytes(bytes) {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

function BatchProgress({ items, currentFileIndex, elapsed }) {
  const done = items.filter((i) => i.status === "done").length;
  const total = items.length;
  const overallPct = total > 0 ? (done / total) * 100 : 0;

  return (
    <div className="rounded-3xl border border-violet-400/15 bg-violet-500/[0.04] p-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin text-violet-300" />
          <span className="text-sm font-semibold text-white">Processing {total} image{total !== 1 ? "s" : ""}...</span>
        </div>
        <div className="flex items-center gap-3 text-xs text-white/40">
          <span>{done}/{total} done</span>
          <span>{elapsed}s</span>
        </div>
      </div>

      <div className="mb-4 h-2 w-full overflow-hidden rounded-full bg-violet-950">
        <div className="h-full rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500 transition-all duration-300" style={{ width: `${overallPct}%` }} />
      </div>

      <div className="space-y-1.5 max-h-[250px] overflow-y-auto">
        {items.map((item, idx) => (
          <div key={idx} className={`flex items-center gap-3 rounded-xl p-2.5 text-sm transition-all ${
            idx === currentFileIndex && (item.status === "processing" || item.status === "downloading") ? "bg-violet-500/10 border border-violet-400/20" :
            item.status === "done" ? "bg-emerald-500/5 border border-emerald-400/10" :
            item.status === "error" ? "bg-red-500/5 border border-red-400/10" :
            "border border-transparent"
          }`}>
            {item.status === "done" ? (
              <CheckCircle className="h-4 w-4 shrink-0 text-emerald-400" />
            ) : item.status === "error" ? (
              <AlertCircle className="h-4 w-4 shrink-0 text-red-400" />
            ) : item.status === "processing" ? (
              <Loader2 className="h-4 w-4 shrink-0 animate-spin text-violet-300" />
            ) : item.status === "downloading" ? (
              <Upload className="h-3.5 w-3.5 shrink-0 animate-pulse text-violet-300" />
            ) : (
              <Circle className="h-4 w-4 shrink-0 text-white/20" />
            )}

            <div className="min-w-0 flex-1">
              <p className="truncate text-xs text-white/80">{item.file.name}</p>
              {item.status === "processing" && item.stage && (
                <p className="mt-0.5 text-[10px] text-violet-300/60">{item.stage} {item.percent ? `(${item.percent}%)` : ""}</p>
              )}
              {item.status === "downloading" && (
                <p className="mt-0.5 text-[10px] text-violet-300/60">
                  {item.bytesRead != null && item.fileSize > 0
                    ? `${formatBytes(item.bytesRead)} / ${formatBytes(item.fileSize)} (${item.transferPct || 0}%)`
                    : item.stage || "Downloading..."}
                </p>
              )}
              {item.status === "error" && <p className="mt-0.5 text-[10px] text-red-300/60">{item.error}</p>}
            </div>

            {item.status === "downloading" && item.fileSize > 0 && (
              <div className="shrink-0 w-16">
                <div className="h-1 w-full overflow-hidden rounded-full bg-violet-950">
                  <div className="h-full rounded-full bg-violet-500 transition-all duration-200" style={{ width: `${item.transferPct || 0}%` }} />
                </div>
              </div>
            )}
            {item.status === "done" && item.elapsed && (
              <span className="shrink-0 text-[10px] text-white/30">{item.elapsed}s</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function CompareViewer({ originalUrl, processedUrl }) {
  const [position, setPosition] = useState(50);
  return (
    <div className="overflow-hidden rounded-3xl border border-white/10 bg-black">
      <div className="@container relative aspect-video overflow-hidden">
        <img src={originalUrl} alt="Original" className="absolute inset-0 h-full w-full object-contain" />
        <div className="absolute inset-y-0 left-0 overflow-hidden" style={{ width: `${position}%` }}>
          <img src={processedUrl} alt="Cinematic" className="absolute inset-y-0 left-0 h-full max-w-none object-contain" style={{ width: "100cqw" }} />
        </div>
        <div className="pointer-events-none absolute left-5 top-5 rounded-full border border-white/10 bg-black/50 px-3 py-1.5 text-xs font-medium text-white backdrop-blur">CINEMATIC</div>
        <div className="pointer-events-none absolute right-5 top-5 rounded-full border border-white/10 bg-black/50 px-3 py-1.5 text-xs font-medium text-white backdrop-blur">ORIGINAL</div>
        <div className="pointer-events-none absolute inset-y-0 w-px bg-white shadow-[0_0_15px_rgba(255,255,255,0.5)]" style={{ left: `${position}%` }}>
          <div className="absolute left-1/2 top-1/2 flex h-11 w-11 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-white/20 bg-black/70 shadow-xl backdrop-blur">
            <ChevronDown className="h-4 w-4 -rotate-90 text-white" />
            <ChevronDown className="h-4 w-4 rotate-90 text-white" />
          </div>
        </div>
        <input type="range" min="0" max="100" value={position} onChange={(e) => setPosition(Number(e.target.value))} className="absolute inset-0 h-full w-full cursor-ew-resize opacity-0" />
      </div>
      <div className="flex items-center justify-between px-5 py-4 text-xs text-white/40">
        <span>Drag the divider to compare</span><span>{position}%</span>
      </div>
    </div>
  );
}

// -- Helpers ----------------------------------------------------------------

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const b64 = reader.result.split(",")[1];
      resolve(b64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function b64ToBlobUrl(b64, mime = "image/jpeg") {
  const byteChars = atob(b64);
  const bytes = new Uint8Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i++) bytes[i] = byteChars.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}

// -- Component --------------------------------------------------------------

export default function App() {
  const [mode, setMode] = useState("automatic");
  const [files, setFiles] = useState([]);
  const [urls, setUrls] = useState("");
  const [inputMode, setInputMode] = useState("files");
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState("");
  const [health, setHealth] = useState("checking");
  const [parameters, setParameters] = useState(CUSTOM_DEFAULTS);
  const [maxSize, setMaxSize] = useState(2048);
  const [concurrency, setConcurrency] = useState(2);

  const [queueItems, setQueueItems] = useState([]);
  const [currentFileIndex, setCurrentFileIndex] = useState(-1);
  const [elapsed, setElapsed] = useState(0);
  const [results, setResults] = useState([]);
  const [selectedResult, setSelectedResult] = useState(null);
  const [batchDone, setBatchDone] = useState(false);

  const timerRef = useRef(null);
  const pollRef = useRef(null);
  const activeJobRef = useRef(null);

  // Wait for pywebview bridge to be ready
  useEffect(() => {
    let cancelled = false;

    function initApi() {
      if (window.pywebview?.api) {
        API = window.pywebview.api;
        if (!cancelled) setHealth("ready");
        return true;
      }
      return false;
    }

    // Already ready
    if (initApi()) return () => { cancelled = true; };

    // Wait for pywebviewready event
    function onReady() { initApi(); }
    window.addEventListener("pywebviewready", onReady);

    // Fallback: poll in case event already fired before listener
    const poll = setInterval(() => { if (initApi()) clearInterval(poll); }, 200);

    return () => { cancelled = true; window.removeEventListener("pywebviewready", onReady); clearInterval(poll); };
  }, []);

  const updateParameter = (key, value) => setParameters((prev) => ({ ...prev, [key]: value }));
  const resetCustom = () => setParameters(CUSTOM_DEFAULTS);

  // -- Build params dict -----------------------------------------------------
  const buildParams = useCallback(() => {
    const p = { max_size: maxSize, concurrency };
    if (mode === "full") Object.assign(p, FULL_POWER);
    if (mode === "custom") Object.assign(p, parameters);
    return p;
  }, [mode, parameters, maxSize, concurrency]);

  // -- Process batch ---------------------------------------------------------
  async function processBatch() {
    if (!API) { setError("pywebview API not available."); return; }

    const isUrlMode = inputMode === "urls";
    const urlList = isUrlMode ? urls.split("\n").map((u) => u.trim()).filter(Boolean) : [];

    if (isUrlMode && urlList.length === 0) { setError("Please enter at least one image URL."); return; }
    if (!isUrlMode && files.length === 0) { setError("Please upload images first."); return; }

    const itemCount = isUrlMode ? urlList.length : files.length;
    setProcessing(true);
    setError("");
    setResults([]);
    setSelectedResult(null);
    setCurrentFileIndex(-1);
    setElapsed(0);
    setBatchDone(false);

    const items = (isUrlMode ? urlList : files).map((f) => ({
      file: isUrlMode ? { name: f.split("/").pop().split("?")[0] || f } : f,
      status: "queued", stage: "", percent: 0, error: "", elapsed: 0, outputUrl: null,
    }));
    setQueueItems(items);

    const startTime = Date.now();
    timerRef.current = setInterval(() => setElapsed(((Date.now() - startTime) / 1000).toFixed(1)), 100);

    try {
      const params = buildParams();
      let batchId;

      if (isUrlMode) {
        const res = await API.start_batch_urls(urlList, params);
        batchId = res.batch_id;
      } else {
        const filesData = [];
        for (const f of files) {
          const data = await readFileAsBase64(f);
          filesData.push({ filename: f.name, data });
        }
        const res = await API.start_batch(filesData, params);
        batchId = res.batch_id;
      }

      activeJobRef.current = batchId;

      // Poll for progress
      pollRef.current = setInterval(async () => {
        try {
          const events = await API.get_progress(batchId);
          for (const data of events) {
            if (data.type === "file_start") {
              setCurrentFileIndex(data.file_index);
              setQueueItems((prev) => prev.map((item, i) => i === data.file_index ? { ...item, status: "processing" } : item));
            } else if (data.type === "file_progress") {
              setQueueItems((prev) => prev.map((item, i) => i === data.file_index ? { ...item, stage: data.stage, percent: data.percent } : item));
            } else if (data.type === "file_done") {
              const outputUrl = b64ToBlobUrl(data.image_base64);

              let originalUrl;
              if (isUrlMode) {
                originalUrl = `${window.location.origin}/original/${batchId}/${encodeURIComponent(data.file_name)}`;
              } else {
                const origFile = files[data.file_index];
                originalUrl = origFile ? URL.createObjectURL(origFile) : null;
              }

              setQueueItems((prev) => prev.map((item, i) => i === data.file_index ? { ...item, status: "done", stage: "done", percent: 100, outputUrl, elapsed: ((Date.now() - startTime) / 1000).toFixed(1) } : item));
              setResults((prev) => [...prev, {
                name: data.file_name,
                original: isUrlMode ? urlList[data.file_index] : files[data.file_index],
                originalUrl, outputUrl, plan: data.plan,
              }]);
            } else if (data.type === "file_error") {
              setQueueItems((prev) => prev.map((item, i) => i === data.file_index ? { ...item, status: "error", error: data.detail } : item));
            } else if (data.type === "batch_done") {
              setBatchDone(true);
              clearInterval(pollRef.current);
              clearInterval(timerRef.current);
              setProcessing(false);
            } else if (data.type === "batch_error") {
              throw new Error(data.detail || "Batch processing failed");
            }
          }
        } catch (e) {
          if (e.message?.includes("Batch processing failed")) {
            setError(e.message);
            clearInterval(pollRef.current);
            clearInterval(timerRef.current);
            setProcessing(false);
          }
        }
      }, 500);

    } catch (err) {
      setError(err?.message || "Processing failed.");
      clearInterval(timerRef.current);
      setProcessing(false);
    }
  }

  // -- Download all ----------------------------------------------------------
  async function downloadAll() {
    if (results.length === 0) return;
    if (results.length === 1) {
      const a = document.createElement("a");
      a.href = results[0].outputUrl;
      a.download = `cinematic-${results[0].name}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      return;
    }
    const zip = new JSZip();
    const folder = zip.folder("cinematic-results");
    for (const r of results) {
      const resp = await fetch(r.outputUrl);
      const blob = await resp.blob();
      folder.file(`cinematic-${r.name}`, blob);
    }
    const zipBlob = await zip.generateAsync({ type: "blob" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(zipBlob);
    a.download = "cinematic-results.zip";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  // -- New batch -------------------------------------------------------------
  function newBatch() {
    setFiles([]);
    setUrls("");
    setQueueItems([]);
    setResults([]);
    setSelectedResult(null);
    setError("");
    setCurrentFileIndex(-1);
    setElapsed(0);
    setBatchDone(false);
    activeJobRef.current = null;
    clearInterval(pollRef.current);
  }

  const itemCount = inputMode === "urls" ? urls.split("\n").map((u) => u.trim()).filter(Boolean).length : files.length;

  const modeDescription = {
    automatic: "Scene-adaptive mode. The engine analyzes each image and dynamically decides the cinematic treatment.",
    full: "Maximum cinematic rendering using your full-power parameter preset.",
    custom: "Manually control the cinematic pipeline with individual parameter sliders.",
  }[mode];

  const groupedParameters = ["Film", "Optical", "Geometry"];

  return (
    <div className="min-h-screen bg-[#07040d] text-white">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute left-1/4 top-[-10rem] h-[30rem] w-[30rem] rounded-full bg-violet-700/10 blur-[120px]" />
        <div className="absolute right-[-5rem] top-1/3 h-[28rem] w-[28rem] rounded-full bg-fuchsia-700/8 blur-[130px]" />
      </div>

      <div className="relative mx-auto max-w-7xl px-5 py-6 sm:px-8 lg:px-10">
        <header className="mb-10 flex flex-col gap-5 border-b border-white/7 pb-7 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-4">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-violet-400/20 bg-violet-500/10 shadow-lg shadow-violet-950/20"><Sparkles className="h-6 w-6 text-violet-300" /></div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight">Cinematic AI</h1>
                <span className="rounded-full border border-violet-400/15 bg-violet-500/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-widest text-violet-300">Desktop</span>
              </div>
              <p className="text-sm text-white/35">Adaptive cinematic photo transformation</p>
            </div>
          </div>
          <div className={`flex items-center gap-2 self-start rounded-full border border-white/8 bg-white/[0.025] px-3 py-1.5 text-xs text-white/60 md:self-auto`}>
            <span className={`h-2 w-2 rounded-full ${health === "ready" ? "bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,.7)]" : health === "offline" ? "bg-red-400" : "animate-pulse bg-yellow-400"}`} />
            {health === "ready" ? "Engine Ready" : health === "offline" ? "Engine Offline" : "Checking..."}
          </div>
        </header>

        <section className="mb-9">
          <div className="max-w-3xl">
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-violet-400/15 bg-violet-500/8 px-3 py-1.5 text-xs font-medium text-violet-300">
              <WandSparkles className="h-3.5 w-3.5" /> Serverless desktop app
            </div>
            <h2 className="text-4xl font-black tracking-tight text-white sm:text-5xl">
              Turn ordinary photos into
              <span className="bg-gradient-to-r from-violet-300 via-fuchsia-300 to-purple-400 bg-clip-text text-transparent"> cinematic images.</span>
            </h2>
            <p className="mt-4 max-w-2xl text-base leading-7 text-white/45">
              No server required. Processing runs locally via pywebview.
            </p>
          </div>
        </section>

        <div className="grid gap-7 lg:grid-cols-[1.3fr_.7fr]">
          <main className="space-y-7">
            <InputSource inputMode={inputMode} setInputMode={setInputMode} files={files} setFiles={setFiles} urls={urls} setUrls={setUrls} />

            <section className="rounded-3xl border border-white/8 bg-white/[0.025] p-5 sm:p-6">
              <div className="mb-5 flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-500/10"><Settings2 className="h-5 w-5 text-violet-300" /></div>
                <div>
                  <h3 className="font-semibold text-white">Processing Mode</h3>
                  <p className="text-sm text-white/35">Choose how the engine should control the render plan.</p>
                </div>
              </div>
              <div className="relative">
                <select value={mode} onChange={(e) => setMode(e.target.value)} className="w-full appearance-none rounded-2xl border border-violet-400/15 bg-[#10091a] px-4 py-4 pr-12 text-sm font-medium text-white outline-none transition hover:border-violet-400/30 focus:border-violet-400/50">
                  <option value="automatic">Automatic - Scene Adaptive</option>
                  <option value="full">Full Power - Maximum Cinematic</option>
                  <option value="custom">Custom - Manual Controls</option>
                </select>
                <ChevronDown className="pointer-events-none absolute right-4 top-1/2 h-5 w-5 -translate-y-1/2 text-violet-300" />
              </div>
              <div className="mt-4 flex items-start gap-3 rounded-2xl border border-violet-400/10 bg-violet-500/[0.04] p-4">
                <Info className="mt-0.5 h-4 w-4 shrink-0 text-violet-300" />
                <p className="text-sm leading-6 text-white/45">{modeDescription}</p>
              </div>
            </section>

            {mode === "custom" && (
              <section className="rounded-3xl border border-white/8 bg-white/[0.025] p-5 sm:p-6">
                <div className="mb-6 flex items-center justify-between gap-4">
                  <div><h3 className="font-semibold text-white">Custom Cinematic Controls</h3><p className="mt-1 text-sm text-white/35">Tune the 13 render parameters directly.</p></div>
                  <button onClick={resetCustom} className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 text-xs text-white/60 transition hover:border-violet-400/20 hover:text-white">
                    <RotateCcw className="h-3.5 w-3.5" /> Reset
                  </button>
                </div>
                <div className="space-y-7">
                  {groupedParameters.map((group) => (
                    <div key={group}>
                      <div className="mb-3 flex items-center gap-2"><div className="h-px flex-1 bg-white/6" /><span className="text-[10px] font-semibold uppercase tracking-[0.22em] text-violet-300/50">{group}</span><div className="h-px flex-1 bg-white/6" /></div>
                      <div className="grid gap-3 md:grid-cols-2">
                        {Object.entries(PARAMETER_INFO).filter(([, info]) => info.group === group).map(([name]) => (
                          <SliderParameter key={name} name={name} value={parameters[name]} onChange={(v) => updateParameter(name, v)} />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            <div className="flex gap-3">
              <button onClick={processBatch} disabled={itemCount === 0 || processing}
                className="group relative flex flex-1 items-center justify-center gap-3 overflow-hidden rounded-2xl border border-violet-400/20 bg-gradient-to-r from-violet-600 to-fuchsia-600 px-5 py-4 font-semibold text-white shadow-xl shadow-violet-950/30 transition hover:scale-[1.005] hover:shadow-violet-900/40 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:scale-100">
                <span className="absolute inset-0 bg-white/0 transition group-hover:bg-white/5" />
                {processing ? (<><Loader2 className="relative h-5 w-5 animate-spin" /><span className="relative">Processing {itemCount} image{itemCount !== 1 ? "s" : ""}...</span></>) :
                (<><Sparkles className="relative h-5 w-5" /><span className="relative">Create Cinematic {itemCount > 0 ? `(${itemCount})` : "Image"}</span></>)}
              </button>
              {(processing || results.length > 0) && (
                <button onClick={newBatch} className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/[0.03] px-5 py-4 text-sm font-medium text-white/70 transition hover:border-violet-400/20 hover:text-white">
                  <RotateCcw className="h-4 w-4" /> New
                </button>
              )}
            </div>

            {error && (
              <div className="flex items-start gap-3 rounded-2xl border border-red-400/15 bg-red-400/5 p-4 text-sm text-red-200">
                <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" /><span>{error}</span>
              </div>
            )}

            {processing && <BatchProgress items={queueItems} currentFileIndex={currentFileIndex} elapsed={elapsed} />}

            {results.length > 0 && (
              <section className="rounded-3xl border border-white/8 bg-white/[0.025] p-5 sm:p-6">
                <div className="mb-5 flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
                  <div>
                    <div className="mb-2 inline-flex items-center gap-2 text-xs font-medium uppercase tracking-widest text-emerald-300">
                      <CheckCircle2 className="h-4 w-4" /> {results.length} image{results.length !== 1 ? "s" : ""} processed
                    </div>
                    <h3 className="text-xl font-bold text-white">Cinematic Results</h3>
                  </div>
                  <div className="flex gap-2">
                    {batchDone && (
                      <button onClick={downloadAll} className="inline-flex items-center gap-2 rounded-xl bg-white px-4 py-2 text-sm font-semibold text-black transition hover:bg-violet-100">
                        <Download className="h-4 w-4" /> Download All
                      </button>
                    )}
                    {!batchDone && (
                      <span className="inline-flex items-center gap-2 rounded-xl border border-violet-400/20 bg-violet-500/10 px-4 py-2 text-sm text-violet-200">
                        <Loader2 className="h-4 w-4 animate-spin" /> Processing...
                      </span>
                    )}
                  </div>
                </div>

                <div className="grid max-h-[400px] grid-cols-2 gap-3 overflow-y-auto sm:grid-cols-3 md:grid-cols-4">
                  {results.map((r, i) => (
                    <button key={i} onClick={() => setSelectedResult(selectedResult === i ? null : i)}
                      className={`group relative overflow-hidden rounded-xl border transition-all ${selectedResult === i ? "border-violet-400/40 ring-2 ring-violet-400/20" : "border-white/8 hover:border-violet-400/20"}`}>
                      <img src={r.outputUrl} alt={r.name} className="aspect-square w-full object-cover" />
                      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent px-2 py-1.5">
                        <p className="truncate text-[10px] text-white/70">{r.name}</p>
                      </div>
                    </button>
                  ))}
                </div>

                {selectedResult !== null && results[selectedResult] && (
                  <div className="mt-5">
                    <p className="mb-3 text-sm text-white/40">Comparing: {results[selectedResult].name}</p>
                    <CompareViewer
                      originalUrl={results[selectedResult].originalUrl || (typeof results[selectedResult].original === "string" ? results[selectedResult].original : URL.createObjectURL(results[selectedResult].original))}
                      processedUrl={results[selectedResult].outputUrl}
                    />
                  </div>
                )}
              </section>
            )}
          </main>

          <aside className="space-y-5">
            <section className="rounded-3xl border border-white/8 bg-white/[0.025] p-5">
              <div className="mb-5 flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-500/10"><ScanSearch className="h-5 w-5 text-violet-300" /></div>
                <div><h3 className="font-semibold text-white">Engine Pipeline</h3><p className="text-xs text-white/30">Local processing</p></div>
              </div>
              <div className="space-y-2">
                {[
                  ["01", "Scene Analysis", "brightness / contrast / warmth"],
                  ["02", "Adaptive 3D LUT", "AI color transformation"],
                  ["03", "Film Transform", "response / rolloff / tint"],
                  ["04", "Optical Renderer", "halation / bloom / grain"],
                  ["05", "Geometry Renderer", "lens / distortion / softness"],
                ].map(([num, title, detail]) => (
                  <div key={num} className="flex gap-3 rounded-2xl border border-white/6 bg-black/10 p-3">
                    <div className="font-mono text-xs text-violet-300/50">{num}</div>
                    <div><div className="text-sm font-medium text-white">{title}</div><div className="mt-0.5 text-[11px] text-white/30">{detail}</div></div>
                  </div>
                ))}
              </div>
            </section>

            <section className="rounded-3xl border border-white/8 bg-white/[0.025] p-5">
              <div className="mb-3 flex items-center gap-2"><SlidersHorizontal className="h-4 w-4 text-violet-300" /><span className="text-sm font-semibold text-white">Max Image Size</span></div>
              <p className="mb-3 text-xs text-white/35">Resize large/DNG images before processing. Lower = faster.</p>
              <div className="flex items-center gap-3">
                <input type="range" min="512" max="8192" step="256" value={maxSize} onChange={(e) => setMaxSize(Number(e.target.value))} className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-violet-950 accent-violet-500" />
                <span className="w-16 text-right font-mono text-xs text-violet-200">{maxSize}px</span>
              </div>
              <div className="mt-2 flex justify-between text-[10px] text-white/25">
                <span>512 (fast)</span><span>8192 (full)</span>
              </div>
            </section>

            <section className="rounded-3xl border border-white/8 bg-white/[0.025] p-5">
              <div className="mb-3 flex items-center gap-2"><Settings2 className="h-4 w-4 text-violet-300" /><span className="text-sm font-semibold text-white">Concurrency</span></div>
              <p className="mb-3 text-xs text-white/35">Process multiple files simultaneously. Higher = faster batch.</p>
              <div className="flex items-center gap-3">
                <input type="range" min="1" max="6" step="1" value={concurrency} onChange={(e) => setConcurrency(Number(e.target.value))} className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-violet-950 accent-violet-500" />
                <span className="w-12 text-right font-mono text-xs text-violet-200">{concurrency}x</span>
              </div>
              <div className="mt-2 flex justify-between text-[10px] text-white/25">
                <span>1 (safe)</span><span>6 (fast)</span>
              </div>
            </section>

            <section className="rounded-3xl border border-violet-400/10 bg-gradient-to-br from-violet-500/8 to-fuchsia-500/5 p-5">
              <div className="mb-3 flex items-center gap-2"><Zap className="h-4 w-4 text-violet-300" /><span className="text-sm font-semibold text-white">Modes</span></div>
              <div className="space-y-3 text-sm">
                <div><div className="font-medium text-violet-200">Automatic</div><p className="mt-1 text-xs leading-5 text-white/35">Each image determines its own treatment.</p></div>
                <div><div className="font-medium text-violet-200">Full Power</div><p className="mt-1 text-xs leading-5 text-white/35">Maximum cinematic preset for all images.</p></div>
                <div><div className="font-medium text-violet-200">Custom</div><p className="mt-1 text-xs leading-5 text-white/35">Same manual settings applied to all images.</p></div>
              </div>
            </section>

            <section className="rounded-3xl border border-white/8 bg-white/[0.025] p-5">
              <div className="mb-3 flex items-center gap-2"><SlidersHorizontal className="h-4 w-4 text-violet-300" /><span className="text-sm font-semibold text-white">Full Power Preset</span></div>
              <div className="space-y-1.5 font-mono text-[11px] text-white/35">
                <div>film = 1.00</div><div>rolloff = 1.00</div><div>shadow tint = 1.00</div><div>color separation = 1.00</div>
                <div>halation = 1.00</div><div>bloom = 1.00</div><div>grain = 0.80</div><div>vignette = 1.00</div>
                <div>lens distortion = 0.50</div><div>chromatic aberration = 0.50</div><div>edge softness = 0.50</div>
              </div>
            </section>
          </aside>
        </div>
      </div>
    </div>
  );
}
