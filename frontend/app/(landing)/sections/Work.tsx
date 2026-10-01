"use client";

import { useEffect, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { ArrowRight } from "lucide-react";
import { resolutionImageUrl, sceneImageUrl, sensorNecessityRenderUrl } from "@/lib/api";

gsap.registerPlugin(ScrollTrigger);

const capabilities = [
  {
    id: 1,
    category: "Visual QA",
    title: "Scene Analysis",
    desc: "Measured Qwen2.5-VL-3B responses over the prepared LoveDA scene with preserved provenance.",
    tag: "0.3 m GSD",
  },
  {
    id: 2,
    category: "Resolution",
    title: "Ladder Eval",
    desc: "5-rung degradation study: 0.3 → 10 m GSD on real LoveDA pixels with per-rung degeneracy guards.",
    tag: "RSVQA-LR",
  },
  {
    id: 3,
    category: "Multi-Sensor",
    title: "Optical + SAR",
    desc: "Real co-gridded Sentinel-2 optical and Sentinel-1 VV/VH observations from the frozen Sensor Necessity scene.",
    tag: "S1 + S2",
  },
];

const preparedScene = sceneImageUrl("loveda_LoveDA_images_png_0_gsd0.3");
const yangtzeRender = (name: "optical" | "sar") => sensorNecessityRenderUrl(
  `/api/sar/sensor-necessity/cdse-yangtze-jiangsu-20200523/render/${name}`
);

export default function Work() {
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.fromTo(
        ".work-header",
        { opacity: 0, y: 50 },
        {
          opacity: 1,
          y: 0,
          duration: 0.9,
          ease: "expo.out",
          scrollTrigger: { trigger: sectionRef.current, start: "top 80%" },
        }
      );
      gsap.fromTo(
        ".work-card",
        { opacity: 0, y: 70 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          ease: "expo.out",
          stagger: 0.15,
          scrollTrigger: { trigger: ".work-grid", start: "top 80%" },
        }
      );
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section ref={sectionRef} id="work" className="py-24 lg:py-36 bg-white">
      <div className="px-6 lg:px-14">
        <div className="work-header flex flex-col lg:flex-row lg:items-end lg:justify-between mb-16 lg:mb-20">
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-[0.3em] mb-3 font-medium">
              AI · SATELLITE · EVIDENCE
            </p>
            <h2 className="text-[clamp(2.5rem,6vw,5rem)] font-black uppercase leading-none">
              CAPABILITIES
            </h2>
          </div>
          <a
            href="/workspace"
            className="mt-6 lg:mt-0 inline-flex items-center gap-2 text-sm font-bold link-underline-landing group text-gray-800"
          >
            OPEN WORKSPACE
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </a>
        </div>

        <div className="work-grid grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-8">
          {capabilities.map((cap) => (
            <a key={cap.id} href="/workspace" className="work-card group block">
              <div className="relative mb-5 aspect-[4/3] overflow-hidden rounded-lg border border-slate-700/70 bg-slate-950 transition-colors duration-500 group-hover:border-slate-500">
                {cap.id === 1 && (
                  <img
                    src={preparedScene}
                    alt="Verified LoveDA 0.3 metre prepared satellite scene"
                    className="absolute inset-0 size-full object-cover transition-transform duration-700 group-hover:scale-[1.015]"
                  />
                )}
                {cap.id === 2 && (
                  <div className="absolute inset-0 grid grid-cols-2">
                    <div className="relative overflow-hidden">
                      <img src={resolutionImageUrl(0.3)} alt="Verified native LoveDA imagery at 0.3 metre GSD" className="size-full object-cover transition-transform duration-700 group-hover:scale-[1.015]" />
                      <span className="absolute left-3 top-3 border border-white/20 bg-black/65 px-2 py-1 font-mono text-[9px] uppercase tracking-wider text-white/80">0.3 M / Native</span>
                    </div>
                    <div className="relative overflow-hidden border-l border-white/50">
                      <img src={resolutionImageUrl(10)} alt="Verified degraded LoveDA imagery at 10 metre GSD" className="size-full object-cover transition-transform duration-700 group-hover:scale-[1.015]" style={{ imageRendering: "pixelated" }} />
                      <span className="absolute right-3 top-3 border border-white/20 bg-black/65 px-2 py-1 font-mono text-[9px] uppercase tracking-wider text-white/80">10 M / Degraded</span>
                    </div>
                  </div>
                )}
                {cap.id === 3 && (
                  <div className="absolute inset-0 grid grid-cols-2">
                    <div className="relative overflow-hidden">
                      <img src={yangtzeRender("optical")} alt="Jiangsu Sentinel-2 optical render" className="size-full object-cover transition-transform duration-700 group-hover:scale-[1.015]" />
                      <span className="absolute left-3 top-3 border border-white/20 bg-black/65 px-2 py-1 font-mono text-[9px] uppercase tracking-wider text-white/80">S2 Optical</span>
                    </div>
                    <div className="relative overflow-hidden border-l border-white/50">
                      <img src={yangtzeRender("sar")} alt="Jiangsu Sentinel-1 SAR render" className="size-full object-cover transition-transform duration-700 group-hover:scale-[1.015]" />
                      <span className="absolute right-3 top-3 border border-white/20 bg-black/65 px-2 py-1 font-mono text-[9px] uppercase tracking-wider text-white/80">S1 SAR</span>
                    </div>
                  </div>
                )}
                <div className="absolute inset-0 bg-gradient-to-t from-black via-black/5 to-black/10" />
                <div
                  className="absolute inset-0 opacity-20"
                  style={{
                    backgroundImage:
                      "linear-gradient(rgba(255,237,215,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(255,237,215,0.08) 1px, transparent 1px)",
                    backgroundSize: "32px 32px",
                  }}
                />
                <div className="absolute inset-x-0 bottom-0 flex items-end justify-between gap-4 p-5">
                  <span className="text-2xl font-black uppercase tracking-tight text-white">
                    {cap.category}
                  </span>
                  <span
                    className="shrink-0 border border-white/20 bg-black/60 px-2.5 py-1 text-[9px] font-bold uppercase tracking-[0.2em] text-slate-200"
                  >
                    {cap.tag}
                  </span>
                </div>
              </div>
              <div>
                <p className="text-xs text-gray-400 uppercase tracking-wider mb-1 font-medium">
                  {cap.category}
                </p>
                <h3 className="text-xl font-bold mb-2 group-hover:opacity-70 transition-opacity">
                  {cap.title}
                </h3>
                <p className="text-sm text-gray-500 leading-relaxed">{cap.desc}</p>
              </div>
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}
