import { MCP_RESPONSE_LIMIT_BYTES } from "./mcp-response-limit.js";

export interface McpHttpTimingOptions {
  /** Whole handshake budget, including the initialized notification. */
  initializationTimeoutMs?: number;
  /** Budget from starting a request through receiving its JSON-RPC reply. */
  requestTimeoutMs?: number;
  /** Maximum gap between response body chunks. */
  bodyIdleTimeoutMs?: number;
}

export const DEFAULT_MCP_HTTP_INITIALIZATION_TIMEOUT_MS = 120_000;
const DEFAULT_REQUEST_TIMEOUT_MS = 120_000;
const DEFAULT_BODY_IDLE_TIMEOUT_MS = 30_000;

export interface McpHttpJsonRpcResponse {
  jsonrpc: "2.0";
  id: number | string | null;
  result?: unknown;
  error?: { code: string | number; message: string; [key: string]: unknown };
}

/** Start releasing a body, but never let a peer's cancellation promise hold a caller. */
export function cancelMcpHttpBody(response: Response): void {
  void response.body?.cancel().catch(() => undefined);
}

function waitForHttp<T>(promise: Promise<T>, signal?: AbortSignal, idleTimeoutMs?: number, message = "MCP response body aborted"): Promise<T> {
  const reason = () => {
    const cause: unknown = signal?.reason;
    return cause instanceof Error && cause.name !== "AbortError" ? cause : new Error(message);
  };
  if (signal?.aborted) {
    // The read/fetch was already started; observe its rejection even though the
    // caller's abort wins. Releasing a reader can reject that pending read.
    void promise.catch(() => undefined);
    return Promise.reject(reason());
  }
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => { cleanup(); reject(reason()); };
    const timer = idleTimeoutMs === undefined ? undefined : setTimeout(() => {
      cleanup();
      reject(new Error(`MCP HTTP response body idle timed out after ${idleTimeoutMs}ms`));
    }, idleTimeoutMs);
    const cleanup = () => { clearTimeout(timer); signal?.removeEventListener("abort", onAbort); };
    signal?.addEventListener("abort", onAbort, { once: true });
    promise.then(value => { cleanup(); resolve(value); }, error => { cleanup(); reject(error); });
  });
}

export async function withMcpHttpDeadline<T>(
  operation: (signal: AbortSignal) => Promise<T>,
  signal?: AbortSignal,
  timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS,
  stage = "request",
): Promise<T> {
  const controller = new AbortController();
  const onAbort = () => controller.abort(signal?.reason);
  signal?.addEventListener("abort", onAbort, { once: true });
  if (signal?.aborted) onAbort();
  const timer = setTimeout(() => controller.abort(new Error(`MCP HTTP ${stage} timed out after ${timeoutMs}ms`)), timeoutMs);
  try {
    return await operation(controller.signal);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
    controller.abort();
  }
}

/** Own the fetch, its body and any response that arrives after an abort/deadline. */
export function fetchMcpHttp<T>(
  endpoint: string,
  init: RequestInit,
  consume: (response: Response, signal: AbortSignal) => Promise<T>,
  signal?: AbortSignal,
  options: McpHttpTimingOptions = {},
  fetchImpl: typeof fetch = fetch,
): Promise<T> {
  return withMcpHttpDeadline(async requestSignal => {
    if (requestSignal.aborted) throw requestSignal.reason;
    const pending = fetchImpl(endpoint, { ...init, signal: requestSignal });
    void pending.then(response => {
      if (requestSignal.aborted) cancelMcpHttpBody(response);
    }, () => undefined);
    const response = await waitForHttp(pending, requestSignal, undefined, "MCP HTTP request cancelled");
    return consume(response, requestSignal);
  }, signal, options.requestTimeoutMs);
}

async function readBody<T>(
  response: Response,
  consume: (text: string) => T | undefined,
  finish: () => T,
  signal?: AbortSignal,
  idleTimeoutMs = DEFAULT_BODY_IDLE_TIMEOUT_MS,
): Promise<T> {
  if (!response.body) return finish();
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let bytes = 0;
  let complete = false;
  try {
    for (;;) {
      const { done, value } = await waitForHttp(reader.read(), signal, idleTimeoutMs);
      if (done) {
        complete = true;
        const result = consume(decoder.decode());
        return result ?? finish();
      }
      bytes += value.byteLength;
      if (bytes > MCP_RESPONSE_LIMIT_BYTES) {
        throw new Error(`MCP response exceeds ${MCP_RESPONSE_LIMIT_BYTES}-byte limit`);
      }
      const result = consume(decoder.decode(value, { stream: true }));
      if (result !== undefined) return result;
    }
  } finally {
    if (!complete) void reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

/** Bounded text for HTTP diagnostics; decoding and cancellation share the streaming reader. */
export function readMcpHttpResponseText(response: Response, signal?: AbortSignal, idleTimeoutMs?: number): Promise<string> {
  const parts: string[] = [];
  return readBody(response, text => { parts.push(text); return undefined; }, () => parts.join(""), signal, idleTimeoutMs);
}

function parseEnvelope(text: string, expectedId: number): McpHttpJsonRpcResponse | undefined {
  const value: unknown = JSON.parse(text);
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("Invalid MCP JSON-RPC envelope");
  }
  const record = value as Record<string, unknown>;
  const hasResult = Object.hasOwn(record, "result");
  const hasError = Object.hasOwn(record, "error");
  const hasId = Object.hasOwn(record, "id");
  if (record.jsonrpc !== "2.0") throw new Error("Invalid MCP JSON-RPC envelope");
  if (Object.hasOwn(record, "method")) {
    if (typeof record.method !== "string" || hasResult || hasError
      || (Object.hasOwn(record, "params") && (typeof record.params !== "object" || record.params === null || Array.isArray(record.params)))
      || (hasId && (typeof record.id !== "string" && typeof record.id !== "number"))
      || (typeof record.id === "number" && !Number.isFinite(record.id))) {
      throw new Error("Invalid MCP JSON-RPC envelope");
    }
    // Server request dispatch is unsupported; preserve the ability to read the
    // outgoing request's later response, even when both directions use the same id.
    return undefined;
  }
  if (!hasId || hasResult === hasError || (record.id !== null && typeof record.id !== "string" && typeof record.id !== "number")
    || (typeof record.id === "number" && !Number.isFinite(record.id))) {
    throw new Error("Invalid MCP JSON-RPC envelope");
  }
  if (hasError) {
    const error = record.error;
    if (typeof error !== "object" || error === null || Array.isArray(error)
      || !("code" in error) || (typeof error.code !== "number" && typeof error.code !== "string")
      || (typeof error.code === "number" && !Number.isFinite(error.code))
      || !("message" in error) || typeof error.message !== "string") {
      throw new Error("Invalid MCP JSON-RPC error envelope");
    }
  }
  return record.id === expectedId ? record as unknown as McpHttpJsonRpcResponse : undefined;
}

/** Read event boundaries incrementally and settle only the requested JSON-RPC id. */
export async function readMcpHttpJsonRpcResponse(
  response: Response, expectedId: number, signal?: AbortSignal, idleTimeoutMs?: number,
): Promise<McpHttpJsonRpcResponse> {
  const missing = () => { throw new Error(`MCP response did not contain a matching JSON-RPC result for id ${expectedId}`); };
  if (!(response.headers.get("Content-Type") ?? "").toLowerCase().includes("text/event-stream")) {
    const text = await readMcpHttpResponseText(response, signal, idleTimeoutMs);
    if (!text.trim()) throw new Error("MCP response was empty.");
    return parseEnvelope(text, expectedId) ?? missing();
  }

  let lineBuffer = "";
  let skipLF = false;
  let dataLines: string[] = [];
  const consumeLine = (line: string): McpHttpJsonRpcResponse | undefined => {
    if (line === "") {
      const data = dataLines.join("\n");
      dataLines = [];
      if (!data || data === "[DONE]") return undefined;
      return parseEnvelope(data, expectedId);
    }
    if (line.startsWith(":")) return undefined;
    const colon = line.indexOf(":");
    const field = colon < 0 ? line : line.slice(0, colon);
    if (field === "data") {
      let value = colon < 0 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      dataLines.push(value);
    }
    return undefined;
  };
  return readBody(response, text => {
    if (skipLF && text.length) {
      if (text.startsWith("\n")) text = text.slice(1);
      skipLF = false;
    }
    const endings = /[\r\n]/g;
    let start = 0;
    let ending: RegExpExecArray | null;
    while ((ending = endings.exec(text))) {
      const line = lineBuffer + text.slice(start, ending.index);
      lineBuffer = "";
      start = ending.index + 1;
      if (ending[0] === "\r") {
        if (text[start] === "\n") { start += 1; endings.lastIndex = start; }
        else if (start === text.length) skipLF = true;
      }
      const result = consumeLine(line);
      if (result) return result;
    }
    lineBuffer += text.slice(start);
    return undefined;
  }, missing, signal, idleTimeoutMs);
}
