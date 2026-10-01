"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { 
  Compass, 
  Layers, 
  Radio, 
  ArrowRight, 
  Play, 
  Pause, 
  ChevronDown,
  Crosshair,
  Satellite
} from "lucide-react";

gsap.registerPlugin(ScrollTrigger);

const TOTAL_FRAMES = 160;

interface Chapter {
  id: string;
  tag: string;
  badge: string;
  title: string;
  highlight: string;
  desc: string;
  telemetry: {
    alt: string;
    velocity: string;
    gsd: string;
    sensor: string;
  };
}

const chapters: Chapter[] = [
  {
    id: "01",
    tag: "MISSION CONTROL // LEO INSERTION",
    badge: "PASS 104 • POLAR SUN-SYNC",
    title: "AUTONOMOUS",
    highlight: "SATELLITE INTELLIGENCE",
    desc: "Multi-band spaceborne telemetry with instant frame-by-frame geospatial analysis and verified grounding chains.",
    telemetry: {
      alt: "518.2 KM",
      velocity: "7.66 KM/S",
      gsd: "15.0 M / PX",
      sensor: "MULTISPECTRAL VNIR"
    }
  },
  {
    id: "02",
    tag: "TARGET LOCK // HIGH-RES ZOOM",
    badge: "SAR L-BAND & SUB-METER OPTICAL",
    title: "SUB-METER",
    highlight: "RESOLUTION ZOOM",
    desc: "Cloud-penetrating Synthetic Aperture Radar coupled with high-fidelity optical streams for all-weather clarity.",
    telemetry: {
      alt: "420.0 KM",
      velocity: "7.68 KM/S",
      gsd: "0.5 M / PX",
      sensor: "SAR INTERFEROMETRIC"
    }
  },
  {
    id: "03",
    tag: "NEURAL ENGINE // AUDIT PROVENANCE",
    badge: "ZERO-HALLUCINATION TRUTH",
    title: "TEMPORAL DELTA",
    highlight: "& EVIDENCE CHAINS",
    desc: "Algorithmic change detection traces verified physical shifts on Earth without probabilistic hallucination.",
    telemetry: {
      alt: "310.5 KM",
      velocity: "7.71 KM/S",
      gsd: "0.3 M / PX",
      sensor: "CHANGE DELTA COHERENCE"
    }
  },
  {
    id: "04",
    tag: "TERMINAL DOCKED // READY FOR FLIGHT",
    badge: "NATURAL LANGUAGE QUERY READY",
    title: "ENTER",
    highlight: "SATQUERY WORKSPACE",
    desc: "Query satellite datasets, inspect spectral bands, and run validated grounding workflows in seconds.",
    telemetry: {
      alt: "ORBITAL DOCK",
      velocity: "LOCKED",
      gsd: "SURFACE READY",
      sensor: "AI QUERY ENGINE ONLINE"
    }
  }
];

export default function Hero() {
  const containerRef     = useRef<HTMLDivElement>(null);
  const stickyRef        = useRef<HTMLDivElement>(null);
  const canvasRef        = useRef<HTMLCanvasElement>(null);
  const imagesRef        = useRef<HTMLImageElement[]>([]);
  const currentFrameRef  = useRef<number>(0);
  const autoPlayTimerRef = useRef<number | null>(null);

  const [progress, setProgress]           = useState(0);
  const [activeChapter, setActiveChapter] = useState(0);
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const [loadedPercent, setLoadedPercent] = useState(0);
  const [isReady, setIsReady]             = useState(false);

  // High-performance canvas drawing helper with cover-fit
  const drawFrame = useCallback((frameIdx: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const clampedIdx = Math.max(0, Math.min(TOTAL_FRAMES - 1, frameIdx));
    let targetImg = imagesRef.current[clampedIdx];

    // If exact frame is still downloading, fall back to closest available frame
    if (!targetImg || !targetImg.complete || targetImg.naturalWidth === 0) {
      for (let offset = 1; offset < 20; offset++) {
        const prev = imagesRef.current[clampedIdx - offset];
        if (prev && prev.complete && prev.naturalWidth > 0) {
          targetImg = prev;
          break;
        }
        const next = imagesRef.current[clampedIdx + offset];
        if (next && next.complete && next.naturalWidth > 0) {
          targetImg = next;
          break;
        }
      }
    }

    if (!targetImg || !targetImg.complete || targetImg.naturalWidth === 0) return;

    const cw = canvas.width;
    const ch = canvas.height;
    const iw = targetImg.naturalWidth;
    const ih = targetImg.naturalHeight;

    const scale = Math.max(cw / iw, ch / ih);
    const nw = iw * scale;
    const nh = ih * scale;
    const nx = (cw - nw) / 2;
    const ny = (ch - nh) / 2;

    ctx.clearRect(0, 0, cw, ch);
    ctx.drawImage(targetImg, nx, ny, nw, nh);
  }, []);

  // Preload all 160 frame images in memory
  useEffect(() => {
    let mounted = true;
    let count = 0;
    const imgs: HTMLImageElement[] = [];

    for (let i = 0; i < TOTAL_FRAMES; i++) {
      const img = new Image();
      const num = String(i + 1).padStart(3, "0");
      img.src = `/frames/frame_${num}.jpg`;

      img.onload = () => {
        if (!mounted) return;
        count++;
        setLoadedPercent(Math.round((count / TOTAL_FRAMES) * 100));

        // Draw first frame as soon as it's ready
        if (i === 0) {
          setIsReady(true);
          drawFrame(0);
        }
        if (count >= 15) {
          setIsReady(true);
        }
      };
      imgs.push(img);
    }

    imagesRef.current = imgs;

    return () => {
      mounted = false;
    };
  }, [drawFrame]);

  // Handle high-DPI canvas resizing
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      drawFrame(currentFrameRef.current);
    };

    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [drawFrame]);

  // GSAP ScrollTrigger setup for zero-lag hardware-accelerated scroll scrubbing
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const ctx = gsap.context(() => {
      ScrollTrigger.create({
        trigger: container,
        start: "top top",
        end: "bottom bottom",
        scrub: 0.05, // Instantaneous 0.05s response - eliminating any video seek lag!
        onUpdate: (self) => {
          const p = self.progress;
          setProgress(p);

          const frameIdx = Math.min(
            TOTAL_FRAMES - 1,
            Math.floor(p * (TOTAL_FRAMES - 1))
          );
          currentFrameRef.current = frameIdx;
          drawFrame(frameIdx);

          const chIndex = Math.min(
            chapters.length - 1,
            Math.floor(p * chapters.length)
          );
          setActiveChapter(chIndex);
        },
      });
    }, container);

    return () => {
      ctx.revert();
    };
  }, [drawFrame]);

  // Auto-play mode loop
  useEffect(() => {
    if (!isAutoPlaying) {
      if (autoPlayTimerRef.current) {
        clearInterval(autoPlayTimerRef.current);
        autoPlayTimerRef.current = null;
      }
      return;
    }

    autoPlayTimerRef.current = window.setInterval(() => {
      const nextFrame = (currentFrameRef.current + 1) % TOTAL_FRAMES;
      currentFrameRef.current = nextFrame;
      const newProgress = nextFrame / (TOTAL_FRAMES - 1);
      setProgress(newProgress);
      drawFrame(nextFrame);

      const chIndex = Math.min(
        chapters.length - 1,
        Math.floor(newProgress * chapters.length)
      );
      setActiveChapter(chIndex);
    }, 62); // ~16 fps smooth playback

    return () => {
      if (autoPlayTimerRef.current) {
        clearInterval(autoPlayTimerRef.current);
      }
    };
  }, [isAutoPlaying, drawFrame]);

  const toggleAutoPlay = () => {
    setIsAutoPlaying((prev) => !prev);
  };

  const jumpToChapter = (idx: number) => {
    if (!containerRef.current) return;
    const totalHeight = containerRef.current.offsetHeight - window.innerHeight;
    const targetScroll = containerRef.current.offsetTop + (idx / (chapters.length - 1)) * totalHeight;
    window.scrollTo({ top: targetScroll, behavior: "smooth" });
  };

  const currentCh = chapters[activeChapter];
  const currentTimeSec = (progress * 10).toFixed(2);

  return (
    <div ref={containerRef} className="relative w-full h-[380vh] bg-black">
      {/* ── Sticky Fullscreen Viewport ────────────────────────────── */}
      <div 
        ref={stickyRef}
        className="sticky top-0 h-screen w-full overflow-hidden bg-black flex flex-col justify-between select-none"
      >
        {/* ── Zero-Lag GPU Canvas Sequence ── */}
        <div className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none">
          <canvas
            ref={canvasRef}
            className="w-full h-full block object-cover"
            style={{
              filter: "brightness(0.68) contrast(1.15) saturate(1.1)",
              transform: `scale(${1 + progress * 0.06}) translate3d(0, 0, 0)`,
              willChange: "transform",
            }}
          />
        </div>

        {/* ── High-Tech Vignette Overlays ── */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/95 via-black/35 to-black/70 pointer-events-none" />
        <div className="absolute inset-0 bg-radial-gradient from-transparent via-black/20 to-black/80 pointer-events-none" />
        
        {/* Cyber grid scanline overlay */}
        <div 
          className="absolute inset-0 opacity-[0.035] pointer-events-none" 
          style={{
            backgroundImage: "linear-gradient(rgba(255, 255, 255, 0.2) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.2) 1px, transparent 1px)",
            backgroundSize: "40px 40px"
          }}
        />

        {/* ── TOP HUD BAR ────────────────────────────────────────────── */}
        <div className="relative z-20 pt-20 px-6 lg:px-12 flex flex-wrap items-center justify-between gap-4 text-xs font-mono text-white/70">
          {/* Mission Tag & Live Reticle */}
          <div className="flex items-center gap-3 bg-black/60 backdrop-blur-md border border-white/10 rounded-full px-4 py-1.5 shadow-lg">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
            <span className="w-2 h-2 rounded-full bg-emerald-400 -ml-5" />
            <span className="tracking-widest uppercase font-semibold text-emerald-400">
              SAT-STREAM LIVE
            </span>
            <span className="text-white/30">•</span>
            <span className="text-white/60 tracking-wider">
              {currentCh.badge}
            </span>
          </div>

          {/* Telemetry Metrics */}
          <div className="hidden md:flex items-center gap-6 bg-black/40 backdrop-blur-md border border-white/10 rounded-full px-5 py-1.5">
            <div className="flex items-center gap-2">
              <Compass className="w-3.5 h-3.5 text-cyan-400" />
              <span>ALT: <strong className="text-white font-mono">{currentCh.telemetry.alt}</strong></span>
            </div>
            <div className="flex items-center gap-2">
              <Radio className="w-3.5 h-3.5 text-indigo-400" />
              <span>VEL: <strong className="text-white font-mono">{currentCh.telemetry.velocity}</strong></span>
            </div>
            <div className="flex items-center gap-2">
              <Layers className="w-3.5 h-3.5 text-emerald-400" />
              <span>GSD: <strong className="text-white font-mono">{currentCh.telemetry.gsd}</strong></span>
            </div>
          </div>

          {/* Playback Control */}
          <div className="flex items-center gap-2">
            <button
              onClick={toggleAutoPlay}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/10 hover:bg-white/20 border border-white/10 text-white text-xs font-sans transition-colors cursor-pointer"
              title={isAutoPlaying ? "Switch to Scroll Scrub" : "Auto-Play Video"}
            >
              {isAutoPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              <span>{isAutoPlaying ? "Auto-Play" : "Scroll Scrub"}</span>
            </button>
          </div>
        </div>

        {/* ── CENTER CINEMATIC STAGE ─────────────────────────────────── */}
        <div className="relative z-20 px-6 lg:px-14 flex-1 flex flex-col justify-center max-w-5xl py-8">
          {/* Chapter Eyebrow */}
          <div className="flex items-center gap-3 mb-4">
            <div className="px-2.5 py-1 rounded bg-cyan-500/20 border border-cyan-500/30 text-cyan-300 font-mono text-[11px] tracking-wider uppercase">
              PHASE {currentCh.id} / 04
            </div>
            <span className="text-white/50 text-xs font-mono tracking-widest uppercase">
              {currentCh.tag}
            </span>
          </div>

          {/* Big Cinematic Heading — Vivid+Co weight 400, scale-driven, prism accent */}
          <div className="overflow-hidden mb-6">
            <h1 className="text-[clamp(2.75rem,8vw,7.5rem)] font-normal uppercase leading-[0.98] tracking-[-0.02em] text-bone">
              <span>{currentCh.title}</span>
              <br />
              <span className="prism-text prism-shimmer">
                {currentCh.highlight}
              </span>
            </h1>
          </div>

          {/* Description */}
          <p className="text-white/75 text-base sm:text-lg max-w-xl font-light leading-relaxed mb-8 drop-shadow">
            {currentCh.desc}
          </p>

          {/* Action CTAs */}
          <div className="flex flex-wrap items-center gap-4">
            <a
              href="/workspace"
              className="group inline-flex items-center gap-2.5 px-6 py-3.5 border border-bone text-bone text-sm uppercase tracking-[0.04em] transition-colors duration-500 hover:bg-bone hover:text-obsidian cursor-pointer"
              style={{ transitionTimingFunction: "cubic-bezier(0.52,0.01,0,1)" }}
            >
              <Satellite className="w-4 h-4" />
              <span>Launch Workspace</span>
              <ArrowRight className="w-4 h-4" />
            </a>
            <a
              href="#work"
              onClick={(e) => {
                e.preventDefault();
                document.querySelector("#work")?.scrollIntoView({ behavior: "smooth" });
              }}
              className="inline-flex items-center gap-2 px-1 py-3.5 text-bone/70 text-sm uppercase tracking-[0.04em] transition-colors duration-500 hover:text-bone cursor-pointer link-underline-landing"
            >
              <span>Explore capabilities ↓</span>
            </a>
          </div>
        </div>

        {/* ── BOTTOM HUD & TIMELINE SCRUBBER ─────────────────────────── */}
        <div className="relative z-20 pb-8 px-6 lg:px-12">
          {/* Chapter Selector & Progress Track */}
          <div className="bg-black/60 backdrop-blur-xl border border-white/10 rounded-2xl p-4 lg:p-5 shadow-2xl">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-3">
              {/* Timeline Label */}
              <div className="flex items-center gap-4 text-xs font-mono">
                <span className="text-cyan-400 font-bold tracking-widest uppercase flex items-center gap-1.5">
                  <Crosshair className="w-3.5 h-3.5" />
                  SCROLL TIMELINE
                </span>
                <span className="text-white/40">|</span>
                <span className="text-white/80">
                  {currentTimeSec}s / 10.00s
                </span>
                <span className="text-white/40">|</span>
                <span className="text-white/60">
                  {(progress * 100).toFixed(0)}% ORBIT PROGRESS
                </span>
              </div>

              {/* Chapter Jump Buttons */}
              <div className="flex items-center gap-2 overflow-x-auto pb-1 md:pb-0">
                {chapters.map((ch, idx) => (
                  <button
                    key={ch.id}
                    onClick={() => jumpToChapter(idx)}
                    className={`px-3 py-1 rounded text-xs font-mono transition-all flex items-center gap-1.5 cursor-pointer ${
                      activeChapter === idx
                        ? "bg-cyan-500 text-black font-bold shadow-sm"
                        : "bg-white/5 hover:bg-white/10 text-white/60 border border-white/10"
                    }`}
                  >
                    <span>{ch.id}</span>
                    <span className="hidden sm:inline">{ch.title.split(" ")[0]}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Glowing Interactive Progress Bar */}
            <div 
              className="relative w-full h-2 bg-white/10 rounded-full overflow-hidden cursor-pointer group"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const clickX = (e.clientX - rect.left) / rect.width;
                if (containerRef.current) {
                  const totalHeight = containerRef.current.offsetHeight - window.innerHeight;
                  const targetScroll = containerRef.current.offsetTop + clickX * totalHeight;
                  window.scrollTo({ top: targetScroll, behavior: "smooth" });
                }
              }}
            >
              <div 
                className="absolute top-0 left-0 bottom-0 bg-gradient-to-r from-cyan-500 via-teal-400 to-indigo-500 rounded-full transition-all duration-75 shadow-[0_0_12px_rgba(6,182,212,0.8)]"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
          </div>

          {/* Scroll Prompt at very start */}
          {progress < 0.08 && (
            <div className="mt-4 flex items-center justify-center gap-2 text-white/50 text-xs font-mono animate-bounce">
              <ChevronDown className="w-4 h-4 text-cyan-400" />
              <span>SCROLL DOWN TO PILOT SATELLITE PASS</span>
              <ChevronDown className="w-4 h-4 text-cyan-400" />
            </div>
          )}
        </div>

        {/* Bottom edge gradient to transition into next section */}
        <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-black to-transparent pointer-events-none" />
      </div>
    </div>
  );
}
