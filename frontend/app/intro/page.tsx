import type { Metadata } from "next";
import { CinematicIntro } from "@/components/cinematic/CinematicIntro";

export const metadata: Metadata = { title: "Introduction" };

export default function IntroPage() {
  return <CinematicIntro />;
}
