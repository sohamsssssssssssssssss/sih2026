import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryPanel } from "./QueryPanel";
import { GOLDEN_OPTION, type SceneOption } from "@/lib/scenes";
import { GOLDEN_SCENE } from "@/lib/workspace-contract";

const REPLAY = /Show committed result \(cached replay\)/;

function json(status: number, body: unknown) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

function cachedResult() {
  return {
    answer: "Yes",
    execution_mode: "cached_result",
    results_artifact: "results/ladder.json",
    model: { name: "qwen2.5vl-3b", version: "Qwen/Qwen2.5-VL-3B-Instruct" },
    notice: "Cached replay requested; showing the exact committed result for this scene and question. No live inference ran.",
    trace: {
      model_name: "qwen2.5vl-3b",
      model_version: "Qwen/Qwen2.5-VL-3B-Instruct",
      params: { scene_id: GOLDEN_SCENE.id, execution_mode: "cached_result", results_artifact: "results/ladder.json" },
      input_summary: { question: GOLDEN_SCENE.question as string, image_paths: [] as string[], n_images: 1 },
      timestamp_iso: "2026-09-07T00:00:00Z",
      record_hash: "a".repeat(64),
      prev_hash: "",
    },
  };
}

const providerNotReady = { detail: { capability: "single_image_vqa", provider: "qwen2.5vl-3b", reason_code: "CUDA_UNAVAILABLE", detail: "A CUDA GPU is required for live inference." } };

const upload: SceneOption = {
  ...GOLDEN_OPTION, id: "scene_0123456789abcdef0123456789abcdef", label: "t1.tif", kind: "upload",
  suggestedQuestion: null, requestSensor: null, source: "Uploaded scene", format: "TIFF",
};

const bodies = (mock: ReturnType<typeof vi.fn>) => mock.mock.calls.map(([, init]) => JSON.parse((init as RequestInit).body as string));
const ask = () => fireEvent.click(screen.getByRole("button", { name: /ask satquery/i }));

afterEach(() => { vi.unstubAllGlobals(); });

describe("QueryPanel cached replay", () => {
  it("offers cached replay only after a live 503 on the golden pair, and sends cached_result on click", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(json(503, providerNotReady))
      .mockResolvedValueOnce(json(200, cachedResult()));
    vi.stubGlobal("fetch", fetchMock);
    render(<QueryPanel scene={GOLDEN_OPTION} />);
    expect(screen.queryByRole("button", { name: REPLAY })).toBeNull();

    ask();
    const replay = await screen.findByRole("button", { name: REPLAY });
    // The live failure is shown and nothing was retried automatically.
    expect(screen.getByRole("alert")).toHaveTextContent("CUDA_UNAVAILABLE");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(bodies(fetchMock)[0]).toMatchObject({ scene_id: GOLDEN_SCENE.id, question: GOLDEN_SCENE.question, execution_mode: "live" });

    fireEvent.click(replay);
    expect(await screen.findByText("VERIFIED CACHED RESULT")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(bodies(fetchMock)[1]).toMatchObject({ scene_id: GOLDEN_SCENE.id, question: GOLDEN_SCENE.question, execution_mode: "cached_result" });
    expect(screen.getByText(/No live inference ran/)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("does not offer replay when the golden question was edited", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(503, providerNotReady)));
    render(<QueryPanel scene={GOLDEN_OPTION} />);
    fireEvent.change(screen.getByLabelText("Question"), { target: { value: "Is there a road in this image?" } });
    ask();
    await screen.findByRole("alert");
    expect(screen.queryByRole("button", { name: REPLAY })).toBeNull();
  });

  it("does not offer replay for a non-golden scene", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(503, providerNotReady)));
    render(<QueryPanel scene={upload} />);
    fireEvent.change(screen.getByLabelText("Question"), { target: { value: GOLDEN_SCENE.question } });
    ask();
    await screen.findByRole("alert");
    expect(screen.queryByRole("button", { name: REPLAY })).toBeNull();
  });

  it("does not offer replay for non-503 failures on the golden pair", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(502, { detail: "Model execution failed." })));
    render(<QueryPanel scene={GOLDEN_OPTION} />);
    ask();
    await screen.findByRole("alert");
    expect(screen.queryByRole("button", { name: REPLAY })).toBeNull();
  });
});

describe("QueryPanel error details", () => {
  it("renders provider-not-ready details readably", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(503, { detail: { capability: "grounding", provider: "grounding-dino", reason_code: "WEIGHTS_MISSING", detail: "Checkpoint is not cached locally." } })));
    render(<QueryPanel scene={upload} />);
    fireEvent.change(screen.getByLabelText("Question"), { target: { value: "Locate a ship." } });
    ask();
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("grounding capability is not ready");
    expect(alert).toHaveTextContent("grounding-dino");
    expect(alert).toHaveTextContent("WEIGHTS_MISSING");
    expect(alert).toHaveTextContent("Checkpoint is not cached locally.");
    expect(alert).not.toHaveTextContent("[object Object]");
  });

  it("lists failed pair checks and sends the second scene", async () => {
    const fetchMock = vi.fn().mockResolvedValue(json(422, { detail: {
      eligible: false, requested_workflow: "change_vqa",
      failed_checks: [{ code: "acquisition_time_missing", message: "Both scenes need acquisition timestamps." }],
      warnings: [{ code: "pair_group_missing", message: "One or both pair/group identifiers are undeclared." }],
      unknown_metadata: ["scene_2.acquisition_time"],
    } }));
    vi.stubGlobal("fetch", fetchMock);
    const second: SceneOption = { ...upload, id: "scene_fedcba9876543210fedcba9876543210", label: "t2.tif" };
    const onSecond = vi.fn();
    render(<QueryPanel scene={upload} scenes={[upload, second]} onSecondScene={onSecond} />);
    fireEvent.change(screen.getByLabelText(/Second scene/), { target: { value: second.id } });
    expect(onSecond).toHaveBeenCalledWith(second.id);
    fireEvent.click(screen.getByRole("button", { name: "What changed between these images?" }));
    ask();
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("not eligible for change_vqa");
    expect(alert).toHaveTextContent("Both scenes need acquisition timestamps.");
    expect(alert).toHaveTextContent("acquisition_time_missing");
    expect(alert).toHaveTextContent("One or both pair/group identifiers are undeclared.");
    expect(bodies(fetchMock)[0]).toMatchObject({ scene_id: upload.id, scene_id_2: second.id, question: "What changed between these images?", execution_mode: "live" });
  });

  it("shows string details verbatim but hides 500 internals", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(422, { detail: "This request requires two scenes." })));
    const { unmount } = render(<QueryPanel scene={GOLDEN_OPTION} />);
    ask();
    expect(await screen.findByRole("alert")).toHaveTextContent("This request requires two scenes.");
    unmount();

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(500, { detail: "Traceback CUDA /private/secrets" })));
    render(<QueryPanel scene={GOLDEN_OPTION} />);
    ask();
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/Analysis service unavailable/);
    expect(alert).not.toHaveTextContent(/Traceback|private/);
  });

  it("renders grounding evidence returned by a live result", async () => {
    const live = cachedResult();
    Object.assign(live, {
      answer: "Found 1 match for 'ship'.", execution_mode: "live", results_artifact: null, notice: "Live grounding-dino inference completed.",
      model: { name: "grounding-dino", version: "IDEA-Research/grounding-dino-tiny" },
      evidence: [{ type: "bounding_box", label: "ship", coordinates: [0.1, 0.2, 0.3, 0.4], coordinate_space: "normalized_xyxy", confidence: 0.8765, source_scene_id: null }],
    });
    live.trace = { ...live.trace, model_name: "grounding-dino", model_version: "IDEA-Research/grounding-dino-tiny", params: { scene_id: upload.id, execution_mode: "live" } as never, input_summary: { ...live.trace.input_summary, question: "Locate a ship." } };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(200, live)));
    const onResult = vi.fn();
    render(<QueryPanel scene={upload} onResult={onResult} />);
    fireEvent.change(screen.getByLabelText("Question"), { target: { value: "Locate a ship." } });
    ask();
    expect(await screen.findByText("LIVE INFERENCE")).toBeInTheDocument();
    const evidence = screen.getByRole("region", { name: "Supporting evidence" });
    expect(evidence).toHaveTextContent("ship");
    expect(evidence).toHaveTextContent("score 0.8765");
    expect(evidence).toHaveTextContent("normalized_xyxy [0.1, 0.2, 0.3, 0.4]");
    await waitFor(() => expect(onResult).toHaveBeenLastCalledWith(expect.objectContaining({ execution_mode: "live" })));
  });
});
