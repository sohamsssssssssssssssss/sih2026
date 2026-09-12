import { ShieldCheck, ShieldAlert, ShieldQuestion } from "lucide-react";
import { integrityState } from "@/lib/workspace-contract";

export function IntegrityBadge({ verified }: { verified?: boolean | null }) {
  const state = integrityState(verified);
  const Icon = state === "verified" ? ShieldCheck : state === "failed" ? ShieldAlert : ShieldQuestion;
  return <span className={`inline-flex items-center gap-1.5 text-xs font-medium ${state === "verified" ? "text-success" : state === "failed" ? "text-error" : "text-subtitle"}`}><Icon size={14} />{state === "verified" ? "Chain verified" : state === "failed" ? "Verification failed" : "Not checked / Unknown"}</span>;
}
