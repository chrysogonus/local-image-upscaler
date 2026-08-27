import type { Capabilities, JobSettings, JobSnapshot } from "./types";

const API = "/api/v1";

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string | Array<{ msg: string }> };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((item) => item.msg).join("; ");
  } catch {
    // Fall through to the status text.
  }
  return `${response.status} ${response.statusText}`;
}

async function checked(response: Response): Promise<Response> {
  if (!response.ok) throw new Error(await errorMessage(response));
  return response;
}

export async function getCapabilities(): Promise<Capabilities> {
  return (await (await checked(await fetch(`${API}/capabilities`))).json()) as Capabilities;
}

export async function createJob(file: File, settings: JobSettings): Promise<JobSnapshot> {
  const body = new FormData();
  body.append("file", file, file.name);
  Object.entries(settings).forEach(([key, value]) => {
    // An unset optional setting is left out entirely. Appending it would send
    // the string "null", which the backend would read as a real value.
    if (value === null || value === undefined) return;
    body.append(key, String(value));
  });
  return (await (
    await checked(await fetch(`${API}/jobs`, { method: "POST", body }))
  ).json()) as JobSnapshot;
}

export function observeJob(
  jobId: string,
  onJob: (job: JobSnapshot) => void,
  onError: (message: string) => void,
): () => void {
  const source = new EventSource(`${API}/jobs/${jobId}/events`);
  source.addEventListener("job", (event) => {
    const snapshot = JSON.parse((event as MessageEvent<string>).data) as JobSnapshot;
    onJob(snapshot);
    if (["completed", "failed", "cancelled"].includes(snapshot.state)) source.close();
  });
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) return;
    onError("The progress stream was interrupted. The job may still be running locally.");
  };
  return () => source.close();
}

export async function cancelOrDeleteJob(jobId: string): Promise<void> {
  const response = await fetch(`${API}/jobs/${jobId}`, { method: "DELETE" });
  if (!response.ok && response.status !== 404) throw new Error(await errorMessage(response));
}

export async function fetchResult(jobId: string): Promise<Blob> {
  return (await checked(await fetch(`${API}/jobs/${jobId}/result`))).blob();
}
