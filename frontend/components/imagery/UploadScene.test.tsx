import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { UploadScene, toIsoWithZone, validateUploadFile } from "./UploadScene";

function file(name: string, type: string, size = 16) {
  const value = new File([new Uint8Array(Math.min(size, 16))], name, { type });
  if (size > 16) Object.defineProperty(value, "size", { value: size });
  return value;
}

function choose(value: File) {
  fireEvent.change(screen.getByTestId("upload-input"), { target: { files: [value] } });
}

const created = {
  scene_id: "scene_0123456789abcdef0123456789abcdef", filename: "t1.tif", format: "TIFF", width: 64, height: 32,
  sensor: "Sentinel-2", gsd: null, location: null, acquisition_date: "2024-05-01T10:30:00+00:00",
};

afterEach(() => { vi.unstubAllGlobals(); });

describe("UploadScene validation", () => {
  it("rejects unsupported file types before any request", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    render(<UploadScene />);
    choose(file("notes.pdf", "application/pdf"));
    expect(screen.getByRole("alert")).toHaveTextContent(/Unsupported file type/);
    expect(screen.getByRole("button", { name: /upload scene/i })).toBeDisabled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects files over 20 MiB", () => {
    render(<UploadScene />);
    choose(file("big.tif", "image/tiff", 20 * 1024 * 1024 + 1));
    expect(screen.getByRole("alert")).toHaveTextContent(/upload limit is 20 MiB/);
    expect(screen.getByRole("button", { name: /upload scene/i })).toBeDisabled();
  });

  it("accepts TIFFs with an empty MIME type and exactly 20 MiB", () => {
    expect(validateUploadFile(file("scene.TIFF", "", 20 * 1024 * 1024))).toBeNull();
    expect(validateUploadFile(file("scene.jpg", "image/jpeg"))).toBeNull();
    expect(validateUploadFile(file("scene.png", "image/png", 0))).toMatch(/empty/);
  });

  it("produces ISO 8601 timestamps with an explicit zone", () => {
    expect(toIsoWithZone("2024-05-01T10:30", "utc")).toBe("2024-05-01T10:30:00Z");
    expect(toIsoWithZone("2024-05-01T10:30:15", "local")).toMatch(/^2024-05-01T10:30:15[+-]\d{2}:\d{2}$/);
    expect(toIsoWithZone("yesterday", "utc")).toBeNull();
  });
});

describe("UploadScene submission", () => {
  it("posts multipart data with metadata and reports the new scene", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(created), { status: 201 }));
    vi.stubGlobal("fetch", fetchMock);
    const onUploaded = vi.fn();
    render(<UploadScene onUploaded={onUploaded} />);
    choose(file("t1.tif", "image/tiff"));
    fireEvent.change(screen.getByLabelText("Modality"), { target: { value: "optical" } });
    fireEvent.change(screen.getByLabelText("Sensor"), { target: { value: " Sentinel-2 " } });
    fireEvent.change(screen.getByLabelText("Acquisition time"), { target: { value: "2024-05-01T10:30" } });
    expect(screen.getByTestId("timestamp-preview")).toHaveTextContent("Sent as 2024-05-01T10:30:00Z");
    fireEvent.click(screen.getByRole("button", { name: /upload scene/i }));

    expect(await screen.findByRole("button", { name: /uploading/i })).toBeDisabled();
    await waitFor(() => expect(onUploaded).toHaveBeenCalledTimes(1));

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/api/scenes");
    expect(init.method).toBe("POST");
    expect(init.headers).toBeUndefined();
    const body = init.body as FormData;
    expect((body.get("file") as File).name).toBe("t1.tif");
    expect(body.get("modality")).toBe("optical");
    expect(body.get("sensor")).toBe("Sentinel-2");
    expect(body.get("acquisition_timestamp")).toBe("2024-05-01T10:30:00Z");
    expect(body.has("polarization")).toBe(false);
    expect(onUploaded.mock.calls[0][0]).toEqual(created);
    expect(onUploaded.mock.calls[0][1]).toEqual({ modality: "optical", sensor: "Sentinel-2", acquisition_timestamp: "2024-05-01T10:30:00Z" });
    expect(screen.getByRole("status")).toHaveTextContent(/Uploaded t1.tif/);
  });

  it("surfaces the server's 422 detail verbatim", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Acquisition timestamp must include a timezone." }), { status: 422 })));
    const onUploaded = vi.fn();
    render(<UploadScene onUploaded={onUploaded} />);
    choose(file("t1.png", "image/png"));
    fireEvent.click(screen.getByRole("button", { name: /upload scene/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Acquisition timestamp must include a timezone.");
    expect(onUploaded).not.toHaveBeenCalled();
  });

  it("reports a 413 from the server", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "The uploaded image exceeds the 20 MiB limit." }), { status: 413 })));
    render(<UploadScene />);
    choose(file("t1.png", "image/png"));
    fireEvent.click(screen.getByRole("button", { name: /upload scene/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("The uploaded image exceeds the 20 MiB limit.");
  });
});
