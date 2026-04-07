const API_BASE = "http://localhost:8000";

export interface RemoteMidiFile {
  uri: string;
  ticks_per_beat: number;
  content_hash: string;
}

export interface RemoteAudioFile {
  uri: string;
  format: string;
  sample_rate: number;
  duration_sec: number;
  channels: number;
  content_hash: string;
}

export interface JobSummary {
  job_id: string;
  status: "pending" | "running" | "succeeded" | "failed" | "cancelled";
  variant: string;
  title: string | null;
  artist: string | null;
  error: string | null;
  result: Record<string, unknown> | null;
}

export interface SubmitJobParams {
  audio?: RemoteAudioFile;
  midi?: RemoteMidiFile;
  title?: string;
  artist?: string;
  skip_humanizer?: boolean;
  difficulty?: string;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export async function uploadMidi(file: File): Promise<RemoteMidiFile> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/v1/uploads/midi`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<RemoteMidiFile>(res);
}

export async function submitJob(params: SubmitJobParams): Promise<JobSummary> {
  const res = await fetch(`${API_BASE}/v1/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return handleResponse<JobSummary>(res);
}

export async function listJobs(): Promise<JobSummary[]> {
  const res = await fetch(`${API_BASE}/v1/jobs`);
  return handleResponse<JobSummary[]>(res);
}

export async function getJob(jobId: string): Promise<JobSummary> {
  const res = await fetch(`${API_BASE}/v1/jobs/${jobId}`);
  return handleResponse<JobSummary>(res);
}
