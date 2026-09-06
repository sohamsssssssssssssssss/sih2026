import type { Metadata } from "next";
import { CinematicExperience } from "@/components/cinematic/CinematicExperience";

export const metadata: Metadata = {
  title: "Ask Earth a Question",
  description: "Descend from orbit into SatQuery AI's evidence-backed geospatial intelligence workspace.",
};

export default function Home() {
  return <CinematicExperience />;
}
