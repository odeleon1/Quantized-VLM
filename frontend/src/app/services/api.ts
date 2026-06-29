const BASE = import.meta.env.VITE_API_URL ?? "";

function token(): string | null {
  return sessionStorage.getItem("vlmedge_token");
}

function authHeader(): Record<string, string> {
  const t = token();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

function handle401(res: Response) {
  if (res.status === 401) window.dispatchEvent(new Event("vlmedge:unauthorized"));
}

async function post<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.href);
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, String(v)));
  }
  const res = await fetch(url.toString(), { method: "POST", headers: authHeader() });
  if (!res.ok) {
    handle401(res);
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

async function postJson<T>(path: string, body: object): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    handle401(res);
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: authHeader() });
  if (!res.ok) {
    handle401(res);
    throw new Error(res.statusText);
  }
  return res.json() as Promise<T>;
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface InferenceResult {
  text: string;
  tokens: number;
  elapsed_s: number;
  timestamp: string;
}

export interface StatusResult {
  model_ready: boolean;
  camera_ready: boolean;
  recording: boolean;
  recording_file: string | null;
  autoscan: boolean;
  autoscan_interval_s: number;
  memory_available_mb: number;
  last_result: (InferenceResult & { prompt: string; source: string }) | null;
}

export interface SnapshotResult {
  saved: string;
  timestamp: string;
}

export interface RecordResult {
  recording: boolean;
  session_file: string;
}

export interface OutputItem {
  id: number;
  type: "analyze" | "inspect" | "snapshot" | "autoscan" | "record" | "flag";
  timestamp: string;
  file_path: string | null;
  prompt: string | null;
  response: string | null;
  tokens: number | null;
  elapsed_s: number | null;
  user_id: number;
  username?: string; // present in admin endpoint responses
}

export interface FlagResult {
  timestamp: string;
  flagged: boolean;
  frame: string;
  last_inference: (InferenceResult & { prompt: string }) | null;
}

export interface EvalResult {
  label: string;
  prompt: string;
  response: string;
  tokens: number;
  latency_s: number;
  frame_url?: string;
}

export interface EvalStatus {
  running: boolean;
  progress: number;
  total: number;
  current_label: string | null;
  results: EvalResult[];
  report_id: string | null;
  error: string | null;
}

export interface EvalReportMeta {
  id: string;
  timestamp: string | null;
  has_report: boolean;
  result_count: number;
  avg_latency_s: number | null;
  is_baseline: boolean;
  legacy?: boolean;
}

export interface EvalStats {
  avg_latency_s: number;
  total_tokens: number;
  fastest: { label: string; latency_s: number };
  slowest: { label: string; latency_s: number };
}

export interface EvalComparison {
  avg_latency_delta: number;
  direction: "faster" | "slower" | "same";
  per_prompt: Record<string, { latency_delta: number; token_delta: number }>;
}

export interface EvalReportDetail {
  report_id: string;
  timestamp: string;
  results: (EvalResult & { frame_url: string })[];
  is_baseline: boolean;
  stats: EvalStats | null;
  comparison: EvalComparison | null;
}

export interface AuthUser {
  id: number;
  username: string;
  email: string;
  is_admin: boolean;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: AuthUser;
}

export interface AdminUser {
  id: number;
  username: string;
  email: string;
  is_admin: boolean;
  created_at: string;
}

// ── API object ────────────────────────────────────────────────────────────────

export const api = {
  // Inference
  analyze:       () => post<InferenceResult>("/analyze"),
  inspect:       () => post<InferenceResult>("/inspect"),
  snapshot:      () => post<SnapshotResult>("/snapshot"),
  recordStart:   () => post<RecordResult>("/record/start"),
  recordStop:    () => post<RecordResult>("/record/stop"),
  autoscanStart: (interval = 10) => post<{ autoscan: boolean; interval_s: number }>("/autoscan/start", { interval }),
  autoscanStop:  () => post<{ autoscan: boolean }>("/autoscan/stop"),
  flag:          () => post<FlagResult>("/flag"),
  getStatus:     () => get<StatusResult>("/status"),
  streamUrl:     () => `${BASE}/stream?token=${token() ?? ""}`,

  // Evaluation
  evalRun:         () => post<{ started: boolean; total: number }>("/eval/run"),
  evalStatus:      () => get<EvalStatus>("/eval/status"),
  evalReports:     () => get<{ reports: EvalReportMeta[] }>("/eval/reports"),
  evalReport:      (id: string) => get<EvalReportDetail>(`/eval/report/${id}`),
  evalSetBaseline: (id: string) => post<{ baseline_set: boolean; report_id: string; timestamp: string }>(`/eval/set-baseline/${id}`),

  // Eval frame images — must use ?token= because <img src> can't send headers
  evalFrameUrl: (path: string) => `${path}?token=${token() ?? ""}`,

  // Auth
  authLogin:  (identifier: string, password: string) =>
    postJson<LoginResponse>("/auth/login", { identifier, password }),
  authSignup: (username: string, email: string, password: string) =>
    postJson<{ message: string; username: string }>("/auth/signup", { username, email, password }),
  authMe:     () => get<AuthUser>("/auth/me"),

  // Admin
  adminGetUsers: () => get<{ users: AdminUser[] }>("/admin/users"),
  adminPromote:  (id: number) => post<{ success: boolean }>(`/admin/users/${id}/promote`),
  adminDemote:   (id: number) => post<{ success: boolean }>(`/admin/users/${id}/demote`),

  // Library
  libraryOutputs:      (type?: string) =>
    get<{ outputs: OutputItem[] }>(`/library/outputs${type ? `?type=${type}` : ""}`),
  libraryAdminOutputs: (type?: string) =>
    get<{ outputs: OutputItem[] }>(`/library/admin/outputs${type ? `?type=${type}` : ""}`),
  libraryViewUrl:      (id: number) => `${BASE}/library/view/${id}?token=${token() ?? ""}`,
  libraryDownloadUrl:  (id: number) => `${BASE}/library/download/${id}?token=${token() ?? ""}`,
};
