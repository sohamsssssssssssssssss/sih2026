"use client";

import { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { ArrowRight } from "lucide-react";
import { resolutionImageUrl } from "@/lib/api";

gsap.registerPlugin(ScrollTrigger);

const performance = [
  { label: "open_accuracy baseline", value: "0.165", width: "16.5%" },
  { label: "Binary accuracy", value: "0.667", width: "66.7%" },
];

const gsdRungs = ["0.3 M", "1.0 M", "2.0 M", "5.0 M", "10.0 M"];

export default function About() {
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.fromTo(
        ".about-content",
        { opacity: 0, y: 60 },
        {
          opacity: 1,
          y: 0,
          duration: 0.9,
          ease: "expo.out",
          scrollTrigger: { trigger: sectionRef.current, start: "top 70%" },
        }
      );
      gsap.fromTo(
        ".stat-item",
        { opacity: 0, y: 30 },
        {
          opacity: 1,
          y: 0,
          duration: 0.7,
          ease: "expo.out",
          stagger: 0.1,
          scrollTrigger: { trigger: ".stats-grid", start: "top 85%" },
        }
      );
      // Image parallax
      ScrollTrigger.create({
        trigger: sectionRef.current,
        start: "top bottom",
        end: "bottom top",
        scrub: 1,
        onUpdate: (self) => {
          gsap.set(".about-bg", { y: self.progress * -80 });
        },
      });
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section ref={sectionRef} id="about" className="py-24 lg:py-36 bg-black text-white overflow-hidden">
      <div className="px-6 lg:px-14">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-24 items-center">
          {/* Left - measured evidence */}
          <div className="relative">
            <div className="about-bg relative min-h-[580px] overflow-hidden border-y border-white/10 bg-[#0b0e11] lg:min-h-[620px]">
              <div
                aria-hidden="true"
                className="absolute inset-y-0 right-0 w-[76%] bg-cover bg-center opacity-20 grayscale"
                style={{
                  backgroundImage: `url(${resolutionImageUrl(0.3)})`,
                  WebkitMaskImage: "linear-gradient(90deg, transparent 0%, black 38%, black 74%, transparent 100%)",
                  maskImage: "linear-gradient(90deg, transparent 0%, black 38%, black 74%, transparent 100%)",
                }}
              />
              <div
                aria-hidden="true"
                className="absolute inset-0 opacity-50"
                style={{
                  backgroundImage:
                    "linear-gradient(rgba(111,135,156,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(111,135,156,0.08) 1px, transparent 1px)",
                  backgroundSize: "44px 44px",
                }}
              />

              <div className="stats-grid relative z-10 flex min-h-[580px] flex-col p-7 lg:min-h-[620px] lg:p-10">
                <div className="flex items-center justify-between border-b border-white/10 pb-4 text-[10px] uppercase tracking-[0.24em] text-white/40">
                  <span className="text-cyan-400/80">Measured evidence</span>
                  <span>LoveDA / 0.3–10 m GSD</span>
                </div>

                <div className="mt-10 space-y-9">
                  {performance.map((stat) => (
                    <div key={stat.label} className="stat-item">
                      <div className="mb-3 flex items-end justify-between gap-6">
                        <p className="text-[10px] uppercase tracking-[0.2em] text-white/45">{stat.label}</p>
                        <p className="font-mono text-3xl tabular-nums text-cyan-400 lg:text-4xl">{stat.value}</p>
                      </div>
                      <div className="relative h-px bg-white/15">
                        <div className="absolute inset-y-[-2px] left-0 bg-cyan-400/90" style={{ width: stat.width }} />
                        <div className="absolute inset-0 bg-[repeating-linear-gradient(90deg,transparent_0,transparent_calc(10%_-_1px),rgba(255,255,255,0.16)_10%)]" />
                      </div>
                      <div className="mt-2 flex justify-between font-mono text-[9px] text-white/25">
                        <span>0.0</span><span>MEASURED SCALE</span><span>1.0</span>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-auto grid grid-cols-1 gap-8 border-t border-white/10 pt-8 sm:grid-cols-[0.8fr_1.2fr]">
                  <div className="stat-item">
                    <div className="flex items-end justify-between">
                      <p className="text-[10px] uppercase tracking-[0.2em] text-white/45">Test questions</p>
                      <p className="font-mono text-3xl tabular-nums text-cyan-400">10K+</p>
                    </div>
                    <div aria-hidden="true" className="mt-4 grid grid-cols-10 gap-1">
                      {Array.from({ length: 10 }).map((_, index) => (
                        <span key={index} className="h-5 border border-cyan-400/30 bg-cyan-400/10" />
                      ))}
                    </div>
                    <p className="mt-2 font-mono text-[9px] uppercase tracking-wider text-white/25">Question volume / 1K units</p>
                  </div>

                  <div className="stat-item">
                    <div className="flex items-end justify-between">
                      <p className="text-[10px] uppercase tracking-[0.2em] text-white/45">GSD rungs evaluated</p>
                      <p className="font-mono text-3xl tabular-nums text-cyan-400">5</p>
                    </div>
                    <div className="relative mt-5 flex justify-between before:absolute before:left-1 before:right-1 before:top-1 before:h-px before:bg-white/15">
                      {gsdRungs.map((rung) => (
                        <div key={rung} className="relative flex flex-col items-center gap-2">
                          <span className="h-2 w-2 border border-cyan-400/80 bg-[#0b0e11]" />
                          <span className="font-mono text-[8px] text-white/40">{rung}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="mt-7 flex justify-between border-t border-white/10 pt-3 font-mono text-[9px] uppercase tracking-[0.14em] text-white/25">
                  <span>Resolution ladder</span>
                  <span>Frozen measured outputs</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right - content */}
          <div className="about-content">
            <p className="text-xs text-gray-500 uppercase tracking-[0.3em] mb-4 font-medium">ABOUT</p>
            <h2 className="text-[clamp(2rem,5vw,4rem)] font-black uppercase leading-none mb-8">
              THE PROJECT
            </h2>
            <p className="text-3xl lg:text-4xl font-light leading-tight mb-8 text-white/80">
              NOT A DEMO.<br />
              A MEASUREMENT.
            </p>
            <p className="text-white/50 text-lg leading-relaxed mb-8 max-w-md">
              SIH 2026 — a multi-agent geospatial VQA system on Qwen2.5-VL-3B with hash-chained audit traces,
              resolution degradation studies, and offline-first inference.
            </p>
            <p className="mb-8 max-w-lg border-y border-white/10 py-3 text-[10px] uppercase tracking-[0.18em] text-white/40">
              VQA&nbsp;&nbsp;/&nbsp;&nbsp;GROUNDING&nbsp;&nbsp;/&nbsp;&nbsp;MULTI-SENSOR&nbsp;&nbsp;/&nbsp;&nbsp;RESOLUTION&nbsp;&nbsp;/&nbsp;&nbsp;TRACE
            </p>
            <a
              href="/workspace"
              className="inline-flex items-center gap-2 text-sm font-bold text-cyan-400 hover:text-white transition-colors group"
            >
              RUN THE APP
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </a>
            <p className="mt-10 font-mono text-[9px] uppercase tracking-[0.14em] text-white/25">
              LoveDA · Qwen2.5-VL-3B · deterministic re-score of frozen outputs
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
