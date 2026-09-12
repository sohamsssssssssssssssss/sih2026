"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { animate } from "animejs";
import { clampProgress, introStory, LOGIN_START } from "@/animation/introStory";
import "./intro.css";

export function CinematicIntro() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const sectionRef = useRef<HTMLElement>(null);
  const targetTimeRef = useRef(0);
  const displayedTimeRef = useRef(0);
  const rafRef = useRef(0);
  const loginRef = useRef<HTMLDivElement>(null);
  const debugRef = useRef<HTMLOutputElement>(null);
  const reducedRef = useRef(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const section = sectionRef.current;
    const video = videoRef.current;
    const login = loginRef.current;
    if (!section || !video || !login) return;
    const media = matchMedia("(prefers-reduced-motion: reduce)");
    const elements = [...section.querySelectorAll<HTMLElement>("[data-story]")];
    const animations = elements.map(element => animate(element, {
      opacity: [0, 1], y: [20, 0], duration: 1000, autoplay: false, ease: "linear",
    }));
    const loginAnimation = animate(login, {
      opacity: [0, 1], y: [42, 0], duration: 1000, autoplay: false, ease: "linear",
    });
    const steps = [...section.querySelectorAll<HTMLElement>("[data-step]")];
    const stepAnimations = steps.map(element => animate(element, {
      opacity: [0, 1], y: [8, 0], duration: 1000, autoplay: false, ease: "linear",
    }));
    let progress = 0;
    let lastScroll = 0;
    let lastFrame = 0;
    let top = 0;
    let distance = 1;

    const paint = () => {
      const value = reducedRef.current || failed ? 1 : progress;
      const entrance = clampProgress((value - LOGIN_START) / (1 - LOGIN_START));
      section.style.setProperty("--login", String(entrance));
      section.style.setProperty("--progress", String(value));
      elements.forEach((element, index) => {
        const chapter = introStory[index];
        const local = (value - chapter.start) / (chapter.end - chapter.start);
        const visibility = local < 0 || local >= 1 ? 0 : Math.min(clampProgress(local / .18), clampProgress((1 - local) / .2));
        animations[index].seek(visibility * 1000);
        element.setAttribute("aria-hidden", String(visibility < .5));
      });
      stepAnimations.forEach((animation, index) => {
        animation.seek(clampProgress((value - .445 - index * .018) / .012) * 1000);
      });
      loginAnimation.seek(entrance * 1000);
      login.inert = entrance < .95;
      login.setAttribute("aria-hidden", String(entrance < .95));
      if (entrance < .95 && login.contains(document.activeElement)) (document.activeElement as HTMLElement)?.blur();
      if (debugRef.current) debugRef.current.textContent = `SCROLL ${value.toFixed(3)}  VIDEO ${video.currentTime.toFixed(2)} / ${(video.duration || 0).toFixed(2)}  SCENE ${value >= LOGIN_START ? "LOGIN" : introStory.find(c => value >= c.start && value < c.end)?.id.toUpperCase() ?? "TRANSITION"}`;
    };

    const tick = (now: number) => {
      rafRef.current = 0;
      const dt = Math.min(50, lastFrame ? now - lastFrame : 16.67);
      lastFrame = now;
      const delta = targetTimeRef.current - displayedTimeRef.current;
      displayedTimeRef.current = now - lastScroll > 110 || reducedRef.current
        ? targetTimeRef.current
        : displayedTimeRef.current + delta * (1 - Math.pow(.84, dt / 16.67));
      if (Number.isFinite(video.duration) && video.duration > 0 && !video.seeking && Math.abs(video.currentTime - displayedTimeRef.current) > .012) {
        video.currentTime = displayedTimeRef.current;
      }
      paint();
      if (Math.abs(targetTimeRef.current - displayedTimeRef.current) > .008) rafRef.current = requestAnimationFrame(tick);
    };
    const schedule = () => { if (!rafRef.current) rafRef.current = requestAnimationFrame(tick); };
    const update = () => {
      progress = clampProgress((window.scrollY - top) / distance);
      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
      targetTimeRef.current = (reducedRef.current || failed ? 1 : progress) * Math.max(0, duration - .045);
      lastScroll = performance.now();
      schedule();
    };
    const measure = () => {
      reducedRef.current = media.matches;
      top = window.scrollY + section.getBoundingClientRect().top;
      distance = Math.max(1, section.offsetHeight - window.innerHeight);
      update();
    };
    const resize = new ResizeObserver(measure);
    resize.observe(section);
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", measure);
    media.addEventListener("change", measure);
    video.addEventListener("loadedmetadata", update);
    video.addEventListener("seeked", schedule);
    measure();
    return () => {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
      resize.disconnect();
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", measure);
      media.removeEventListener("change", measure);
      video.removeEventListener("loadedmetadata", update);
      video.removeEventListener("seeked", schedule);
      animations.forEach(animation => animation.revert());
      stepAnimations.forEach(animation => animation.revert());
      loginAnimation.revert();
    };
  }, [failed]);

  const scrollTo = (end: boolean) => {
    const section = sectionRef.current;
    if (!section) return;
    const top = window.scrollY + section.getBoundingClientRect().top;
    window.scrollTo({ top: top + (end ? Math.max(0, section.offsetHeight - window.innerHeight) : 0), behavior: reducedRef.current ? "instant" : "smooth" });
  };

  return (
    <section ref={sectionRef} className="sq-intro" data-failed={failed} aria-label="SatQuery introduction">
      <noscript><p style={{ padding: 24 }}>JavaScript is required for the cinematic introduction. <a href="/workspace">Enter the SatQuery workspace →</a></p></noscript>
      <div className="sq-sticky">
        <video ref={videoRef} src="/videos/satquery-intro.mp4" muted playsInline preload="auto" aria-hidden="true" onError={() => setFailed(true)} />
        <div className="sq-vignette" aria-hidden="true" />
        <header className="sq-nav"><span>SATQUERY AI</span><button onClick={() => scrollTo(true)}>SKIP INTRO <span aria-hidden="true">→</span></button></header>
        <div className="sq-stories">
          {introStory.map(chapter => <article key={chapter.id} data-story={chapter.id} aria-hidden="true" className={`sq-copy ${chapter.id === "final" ? "sq-centered" : ""}`}>
            <h1>{chapter.title}</h1>
            {"detail" in chapter && <p className={"query" in chapter ? "sq-query" : ""}>{chapter.detail}</p>}
            {"steps" in chapter && <div className="sq-steps">{["UNDERSTAND", "PLAN", "OBSERVE", "ANALYSE", "ANSWER"].map((step, i) => <span key={step} data-step>{i > 0 && <i aria-hidden="true">↓</i>}{step}</span>)}</div>}
          </article>)}
        </div>
        <div ref={loginRef} className="sq-login" inert aria-hidden="true">
          <p className="sq-wordmark">SATQUERY AI</p>
          <h2>Ask Earth a Question.</h2>
          <div className="sq-auth-actions">
            <button
              type="button"
              className="sq-google-btn"
              onClick={() => router.push("/workspace")}
              title="Continue with Google"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" aria-hidden="true">
                <path fill="#EA4335" d="M12 5c1.6 0 3 .6 4.1 1.6l3.1-3.1C17.3 1.8 14.8 1 12 1 7.5 1 3.7 3.6 1.9 7.3l3.7 2.9C6.5 7.4 9 5 12 5z"/>
                <path fill="#4285F4" d="M23.5 12.3c0-.8-.1-1.7-.2-2.3H12v4.6h6.5c-.3 1.5-1.1 2.8-2.4 3.7l3.7 2.9c2.2-2 3.7-5 3.7-8.9z"/>
                <path fill="#FBBC05" d="M5.6 14.8c-.2-.7-.4-1.5-.4-2.3 0-.8.2-1.6.4-2.3L1.9 7.3C.7 9.7 0 12 0 14.5s.7 4.8 1.9 7.2l3.7-2.9z"/>
                <path fill="#34A853" d="M12 23.5c3.2 0 6-1.1 8-3l-3.7-2.9c-1.1.7-2.5 1.2-4.3 1.2-3 0-5.5-2.4-6.4-5.2L1.9 16.5C3.7 20.2 7.5 23.5 12 23.5z"/>
              </svg>
              <span>CONTINUE WITH GOOGLE</span>
            </button>
            <div className="sq-divider"><span>OR</span></div>
          </div>
          <form
            onSubmit={event => {
              event.preventDefault();
              router.push("/workspace");
            }}
          >
            <label htmlFor="intro-email">Email</label>
            <input
              id="intro-email"
              type="email"
              autoComplete="email"
              placeholder="analyst@isro.gov.in"
              defaultValue="analyst@isro.gov.in"
              className="w-full rounded-16 border border-border bg-surface px-4 py-3 text-sm leading-relaxed text-primary outline-none focus:border-primary mb-3"
            />
            <label htmlFor="intro-password">Password</label>
            <input
              id="intro-password"
              type="password"
              autoComplete="current-password"
              placeholder="••••••••••••"
              defaultValue="satquery-demo"
              className="w-full rounded-16 border border-bg-surface bg-surface px-4 py-3 text-sm leading-relaxed text-primary outline-none focus:border-primary mb-4"
            />
            <button type="submit" className="w-full rounded-24 border border-primary/15 bg-primary/10 px-4 py-3 text-sm font-black tracking-wide text-[#031013] transition hover:text-primary focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 disabled:opacity-60">
              ENTER WORKSPACE <span aria-hidden="true">→</span>
            </button>
          </form>
          <div className="sq-footer">
            Vision-Language Intelligence<br />
            for Satellite Imagery
            <span>SIH26167</span>
          </div>
          <button className="sq-replay w-full rounded-24 border border-primary/15 bg-primary/10 px-4 py-3 text-sm font-medium tracking-[0.08em] text-primary mb-4 transition hover:text-primary focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2" onClick={() => scrollTo(false)}>
            Replay introduction ↑
          </button>
        </div>
        <p className="sq-scroll" aria-hidden="true">SCROLL TO EXPLORE ↓</p>
        {failed && <p role="status" className="sq-error">The cinematic video could not load. You can still enter the workspace.</p>}
        {process.env.NODE_ENV === "development" && <output ref={debugRef} className="sq-debug" aria-hidden="true" />}
      </div>
    </section>
  );
}