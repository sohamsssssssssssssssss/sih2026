import type { Metadata } from "next";
import { AskCapabilities } from "@/components/analysis/AskWorkspace";
import { RuntimeStatus } from "@/components/status/RuntimeStatus";

export const metadata: Metadata = { title: "System" };

export default function SystemPage() {
  return (
    <div>
      <h1 className="text-[42px] sm:text-[54px]">Capability and runtime status</h1>
      <div className="mt-6 grid items-start gap-5 lg:grid-cols-[1fr_360px]">
        <AskCapabilities />
        <RuntimeStatus />
      </div>
    </div>
  );
}
