import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import { HttpMcpClient } from "../tools/session-mcp-client.js";
import { ZaiMcpClient } from "../tools/zai-mcp-client.js";
import { MCP_RESPONSE_LIMIT_BYTES } from "../tools/mcp-response-limit.js";
import { readMcpHttpJsonRpcResponse } from "../tools/mcp-http-response.js";

type Adapter = "session" | "zai";
type Request = { method: string; id?: number };
type Timing = { requestTimeoutMs?: number; initializationTimeoutMs?: number; bodyIdleTimeoutMs?: number };
const encoder = new TextEncoder();

async function within<T>(promise: Promise<T>, ms = 500): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([promise, new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error("fixture watchdog expired")), ms);
    })]);
  } finally {
    clearTimeout(timer);
  }
}

async function withClient(
  adapter: Adapter,
  fetchImpl: typeof fetch,
  action: (call: (signal?: AbortSignal) => Promise<unknown>) => Promise<void>,
  timing: Timing = {},
  endpoint = "https://fixture.invalid",
): Promise<void> {
  const originalFetch = globalThis.fetch;
  let session: HttpMcpClient | undefined;
  try {
    if (adapter === "session") {
      globalThis.fetch = fetchImpl;
      session = new HttpMcpClient({ type: "http", name: "fixture", url: endpoint, headers: [] }, timing);
      await action(async signal => {
        await session!.listTools(signal);
        return session!.callTool("read", {}, signal);
      });
    } else {
      const client = new ZaiMcpClient(fetchImpl, { maxPages: 10, ...timing });
      await action(signal => client.callTool({ endpoint, apiKey: "fixture-key", toolName: "read", arguments: {}, signal }));
    }
  } finally {
    await session?.dispose();
    globalThis.fetch = originalFetch;
  }
}

function resultFor(method: string): unknown {
  return method === "tools/list" ? { tools: [{ name: "read" }] }
    : method === "tools/call" ? { content: [{ type: "text", text: "A🌟B" }] }
      : { protocolVersion: "2025-06-18", capabilities: {} };
}

function jsonResponse(request: Request): Response {
  return new Response(JSON.stringify({ jsonrpc: "2.0", id: request.id, result: resultFor(request.method) }), {
    headers: { "Content-Type": "application/json", "MCP-Session-Id": "stream-session" },
  });
}

function fixtureFetch(reply: (request: Request, init: RequestInit) => Response | Promise<Response>): typeof fetch {
  return (async (_url, init) => {
    assert.ok(init);
    if (init.method === "DELETE") return new Response(null, { status: 202 });
    const request = JSON.parse(String(init.body)) as Request;
    if (request.method !== "initialize") assert.equal(new Headers(init.headers).get("MCP-Session-Id"), "stream-session");
    return reply(request, init);
  }) as typeof fetch;
}

for (const adapter of ["session", "zai"] as const) {
  // Treating method-bearing messages as responses would reject the first chunk
  // or mistake a same-number server request for the outgoing tool call's reply.
  for (const prefix of ["string id", "same-number id", "object params"] as const) {
    test(`${adapter} HTTP skips a valid server request with ${prefix} before the matching SSE result`, async () => {
      let chunksRead = 0;
      let cancellations = 0;
      let response: Response | undefined;
      const fetchImpl = fixtureFetch(request => {
        if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
        if (request.method !== "tools/call") return jsonResponse(request);
        const serverRequest = prefix === "object params"
          ? { jsonrpc: "2.0", id: "server-roots", method: "roots/list", params: { _meta: { progressToken: "roots-progress" } } }
          : { jsonrpc: "2.0", id: prefix === "same-number id" ? request.id : "server-ping", method: "ping" };
        const chunks = [
          encoder.encode(`data: ${JSON.stringify(serverRequest)}\n\n`),
          encoder.encode(`data: ${JSON.stringify({ jsonrpc: "2.0", id: request.id, result: resultFor("tools/call") })}\n\n`),
        ];
        response = new Response(new ReadableStream<Uint8Array>({
          pull(controller) {
            if (chunksRead === chunks.length) { controller.close(); return; }
            controller.enqueue(chunks[chunksRead++]);
          },
          cancel() { cancellations += 1; return new Promise<void>(() => {}); },
        }, { highWaterMark: 0 }), { headers: { "Content-Type": "text/event-stream" } });
        return response;
      });
      await withClient(adapter, fetchImpl, async call => {
        assert.deepEqual(await within(call()), { content: [{ type: "text", text: "A🌟B" }] });
        assert.equal(chunksRead, 2, "the matching result must be read after the server request");
        assert.equal(cancellations, 1, "the open response body must be released once");
        assert.equal(response?.body?.locked, false);
      });
    });
  }

  // Blindly skipping objects with a method or id would let malformed request
  // envelopes disappear and incorrectly return the otherwise valid next chunk.
  for (const [name, fields] of [
    ["non-string method", { id: "server-ping", method: 7 }],
    ["null method", { id: "server-ping", method: null }],
    ["null id", { id: null, method: "ping" }],
    ["boolean id", { id: true, method: "ping" }],
    ["object id", { id: {}, method: "ping" }],
    ["array id", { id: [], method: "ping" }],
    ["null params", { id: "server-ping", method: "ping", params: null }],
    ["array params", { id: "server-ping", method: "ping", params: [] }],
    ["string params", { id: "server-ping", method: "ping", params: "invalid" }],
    ["number params", { id: "server-ping", method: "ping", params: 7 }],
    ["boolean params", { id: "server-ping", method: "ping", params: true }],
    ["request with result", { id: "server-ping", method: "ping", result: {} }],
    ["request with error", { id: "server-ping", method: "ping", error: { code: -32000, message: "invalid" } }],
  ] as const) {
    test(`${adapter} HTTP rejects a malformed server request with ${name} before the matching SSE result`, async () => {
      let chunksRead = 0;
      let cancellations = 0;
      let response: Response | undefined;
      const fetchImpl = fixtureFetch(request => {
        if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
        if (request.method !== "tools/call") return jsonResponse(request);
        const chunks = [
          encoder.encode(`data: ${JSON.stringify({ jsonrpc: "2.0", ...fields })}\n\n`),
          encoder.encode(`data: ${JSON.stringify({ jsonrpc: "2.0", id: request.id, result: resultFor("tools/call") })}\n\n`),
        ];
        response = new Response(new ReadableStream<Uint8Array>({
          pull(controller) {
            if (chunksRead === chunks.length) { controller.close(); return; }
            controller.enqueue(chunks[chunksRead++]);
          },
          cancel() { cancellations += 1; return new Promise<void>(() => {}); },
        }, { highWaterMark: 0 }), { headers: { "Content-Type": "text/event-stream" } });
        return response;
      });
      await withClient(adapter, fetchImpl, async call => {
        await assert.rejects(within(call()), /invalid.*JSON-RPC/i);
        assert.equal(chunksRead, 1, "malformed requests must fail before reading the matching result");
        assert.equal(cancellations, 1);
        assert.equal(response?.body?.locked, false);
      });
    });
  }

  test(`${adapter} HTTP rejects a nonfinite server request id before the matching SSE result`, async () => {
    await withClient(adapter, fixtureFetch(request => {
      if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
      if (request.method !== "tools/call") return jsonResponse(request);
      return new Response('data: {"jsonrpc":"2.0","id":1e400,"method":"ping"}\n\n'
        + `data: ${JSON.stringify({ jsonrpc: "2.0", id: request.id, result: resultFor("tools/call") })}\n\n`,
      { headers: { "Content-Type": "text/event-stream" } });
    }), async call => { await assert.rejects(call(), /invalid.*JSON-RPC/i); });
  });

  test(`${adapter} HTTP reconstructs multiline SSE and selects the matching id across split UTF8 chunks`, async () => {
    const fetchImpl = fixtureFetch(request => {
      if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
      const wire = `: keepalive\r\nevent: message\r\ndata: {"jsonrpc":"2.0","method":"notifications/progress","params":{}}\r\n\r\n`
        + `data: {"jsonrpc":"2.0","id":999,"result":{"ignored":true}}\r\n\r\n`
        + `data: {"jsonrpc":"2.0",\r\ndata: "id":${request.id},"result":${JSON.stringify(resultFor(request.method))}}\r\n\r\n`;
      const bytes = encoder.encode(wire);
      let offset = 0;
      return new Response(new ReadableStream<Uint8Array>({
        pull(controller) {
          if (offset === bytes.length) { controller.close(); return; }
          controller.enqueue(bytes.slice(offset, offset + 1));
          offset += 1;
        },
      }), { headers: { "Content-Type": "text/event-stream", "MCP-Session-Id": "stream-session" } });
    });
    await withClient(adapter, fetchImpl, async call => {
      assert.deepEqual(await call(), { content: [{ type: "text", text: "A🌟B" }] });
    });
  });

  test(`${adapter} HTTP settles a native server result while its SSE response remains open`, async () => {
    const server = createServer((request, response) => {
      void (async () => {
        if (request.method === "DELETE") { response.writeHead(202).end(); return; }
        let text = "";
        for await (const chunk of request) text += chunk;
        const message = JSON.parse(text) as Request;
        if (message.method === "notifications/initialized") { response.writeHead(202).end(); return; }
        if (message.method !== "initialize") assert.equal(request.headers["mcp-session-id"], "stream-session");
        response.writeHead(200, { "Content-Type": "text/event-stream", "MCP-Session-Id": "stream-session" });
        response.write(`data: ${JSON.stringify({ jsonrpc: "2.0", id: message.id, result: resultFor(message.method) })}\n\n`);
      })().catch(error => response.destroy(error));
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const address = server.address();
    assert.ok(address && typeof address !== "string");
    const endpoint = `http://127.0.0.1:${address.port}`;
    try {
      await withClient(adapter, globalThis.fetch, async call => {
        assert.deepEqual(await within(call()), { content: [{ type: "text", text: "A🌟B" }] });
      }, {}, endpoint);
    } finally {
      server.closeAllConnections();
      await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
    }
  });

  test(`${adapter} HTTP owns initialized notification bodies without awaiting stalled cancellation`, async () => {
    let cancelled = false;
    const fetchImpl = fixtureFetch(request => {
      if (request.method !== "notifications/initialized") return jsonResponse(request);
      return new Response(new ReadableStream<Uint8Array>({
        start(controller) { controller.enqueue(encoder.encode("unused notification response")); },
        cancel() { cancelled = true; return new Promise<void>(() => {}); },
      }), { status: 202 });
    });
    await withClient(adapter, fetchImpl, async call => {
      assert.deepEqual(await within(call()), { content: [{ type: "text", text: "A🌟B" }] });
      assert.equal(cancelled, true, "the successful notification body must be released");
    });
  });

  test(`${adapter} HTTP does not let reader cancellation delay a matching SSE result`, async () => {
    let cancelled = false;
    const fetchImpl = fixtureFetch(request => {
      if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
      if (request.method !== "tools/call") return jsonResponse(request);
      return new Response(new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode(`data: {"jsonrpc":"2.0","id":${request.id},"result":{"content":[{"type":"text","text":"A🌟B"}]}}\n\n`));
        },
        cancel() { cancelled = true; return new Promise<void>(() => {}); },
      }), { headers: { "Content-Type": "text/event-stream" } });
    });
    await withClient(adapter, fetchImpl, async call => {
      assert.deepEqual(await within(call()), { content: [{ type: "text", text: "A🌟B" }] });
      assert.equal(cancelled, true);
    });
  });

  test(`${adapter} HTTP rejects oversized unterminated SSE even when cancellation never settles`, async () => {
    let cancelled = false;
    const fetchImpl = fixtureFetch(request => {
      if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
      if (request.method !== "tools/call") return jsonResponse(request);
      return new Response(new ReadableStream<Uint8Array>({
        start(controller) { controller.enqueue(encoder.encode("data: " + " ".repeat(MCP_RESPONSE_LIMIT_BYTES))); },
        cancel() { cancelled = true; return new Promise<void>(() => {}); },
      }), { headers: { "Content-Type": "text/event-stream" } });
    });
    await withClient(adapter, fetchImpl, async call => {
      await assert.rejects(within(call()), /MCP response exceeds/);
      assert.equal(cancelled, true);
    });
  });

  for (const envelope of [null, { jsonrpc: "1.0", id: 3, result: {} }, { jsonrpc: "2.0", id: 3, result: {}, error: { code: -32000, message: "invalid" } }]) {
    test(`${adapter} HTTP rejects malformed JSON-RPC envelope ${JSON.stringify(envelope)}`, async () => {
      await withClient(adapter, fixtureFetch(request => {
        if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
        if (request.method !== "tools/call") return jsonResponse(request);
        return new Response(JSON.stringify(envelope), { headers: { "Content-Type": "application/json" } });
      }), async call => { await assert.rejects(call(), /invalid.*JSON-RPC/i); });
    });
  }

  test(`${adapter} HTTP rejects EOF containing only unrelated response ids`, async () => {
    await withClient(adapter, fixtureFetch(request => {
      if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
      if (request.method !== "tools/call") return jsonResponse(request);
      return new Response('data: {"jsonrpc":"2.0","id":999,"result":{"ignored":true}}\n\n', { headers: { "Content-Type": "text/event-stream" } });
    }), async call => { await assert.rejects(call(), /matching.*JSON-RPC/i); });
  });

  test(`${adapter} HTTP bounds a fetch that never returns headers or honors abort`, async () => {
    let requestSignal: AbortSignal | undefined;
    await withClient(adapter, fixtureFetch((request, init) => {
      if (request.method !== "initialize") throw new Error("unexpected follow-up request");
      requestSignal = init.signal ?? undefined;
      return new Promise<Response>(() => {});
    }), async call => {
      await assert.rejects(within(call()), /request.*timed out/i);
      assert.equal(requestSignal?.aborted, true);
    }, { requestTimeoutMs: 25, initializationTimeoutMs: 200 });
  });

  test(`${adapter} HTTP releases a body whose headers arrive after the request deadline`, async () => {
    let cancelled = false;
    let releaseHeaders!: (response: Response) => void;
    await withClient(adapter, fixtureFetch(() => new Promise<Response>(resolve => { releaseHeaders = resolve; })), async call => {
      await assert.rejects(within(call()), /request.*timed out/i);
      releaseHeaders(new Response(new ReadableStream<Uint8Array>({
        cancel() { cancelled = true; return new Promise<void>(() => {}); },
      })));
      await new Promise(resolve => setImmediate(resolve));
      assert.equal(cancelled, true);
    }, { requestTimeoutMs: 25, initializationTimeoutMs: 200 });
  });

  test(`${adapter} HTTP applies one initialization deadline across initialize and initialized`, async () => {
    let notificationSignal: AbortSignal | undefined;
    await withClient(adapter, fixtureFetch(async (request, init) => {
      if (request.method === "initialize") {
        await new Promise(resolve => setTimeout(resolve, 25));
        return jsonResponse(request);
      }
      if (request.method === "notifications/initialized") {
        notificationSignal = init.signal ?? undefined;
        return new Promise<Response>(() => {});
      }
      throw new Error("initialization must not reach discovery");
    }), async call => {
      await assert.rejects(within(call()), /initialization.*timed out/i);
      assert.equal(notificationSignal?.aborted, true);
    }, { initializationTimeoutMs: 50, requestTimeoutMs: 200 });
  });

  for (const errorResponse of [false, true]) {
    test(`${adapter} HTTP bounds ${errorResponse ? "error" : "SSE"} body idle time and stalled cancellation`, async () => {
      let cancelled = false;
      await withClient(adapter, fixtureFetch(request => {
        if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
        if (request.method !== "tools/call") return jsonResponse(request);
        return new Response(new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(encoder.encode(errorResponse ? "partial error diagnostic" : 'data: {"jsonrpc":"2.0","id":999,"result":{}}\n\n'));
          },
          cancel() { cancelled = true; return new Promise<void>(() => {}); },
        }), { status: errorResponse ? 502 : 200, headers: { "Content-Type": errorResponse ? "text/plain" : "text/event-stream" } });
      }), async call => {
        await assert.rejects(within(call()), /body.*idle.*timed out/i);
        assert.equal(cancelled, true);
      }, { bodyIdleTimeoutMs: 25, requestTimeoutMs: 200 });
    });
  }

  test(`${adapter} HTTP applies an overall deadline despite ongoing SSE comments`, async () => {
    let cancelled = false;
    let timer: NodeJS.Timeout | undefined;
    const fetchImpl = fixtureFetch(request => {
      if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
      if (request.method !== "tools/call") return jsonResponse(request);
      return new Response(new ReadableStream<Uint8Array>({
        start(controller) { timer = setInterval(() => controller.enqueue(encoder.encode(": heartbeat\n\n")), 5); },
        cancel() { cancelled = true; clearInterval(timer); },
      }), { headers: { "Content-Type": "text/event-stream" } });
    });
    try {
      await withClient(adapter, fetchImpl, async call => {
        await assert.rejects(within(call()), /request.*timed out/i);
        assert.equal(cancelled, true);
      }, { requestTimeoutMs: 35, bodyIdleTimeoutMs: 100 });
    } finally {
      clearInterval(timer);
    }
  });

  test(`${adapter} HTTP aborts a pending body even when the stream ignores the request signal`, async () => {
    let cancelled = false;
    let bodyStarted!: () => void;
    const started = new Promise<void>(resolve => { bodyStarted = resolve; });
    let requestSignal: AbortSignal | undefined;
    await withClient(adapter, fixtureFetch((request, init) => {
      if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
      if (request.method !== "tools/call") return jsonResponse(request);
      requestSignal = init.signal ?? undefined;
      return new Response(new ReadableStream<Uint8Array>({
        pull() { bodyStarted(); return new Promise<void>(() => {}); },
        cancel() { cancelled = true; return new Promise<void>(() => {}); },
      }, { highWaterMark: 0 }), { headers: { "Content-Type": "text/event-stream" } });
    }), async call => {
      const controller = new AbortController();
      const pending = call(controller.signal);
      await started;
      controller.abort();
      await assert.rejects(within(pending), /aborted|cancelled/i);
      assert.equal(cancelled, true);
      assert.equal(requestSignal?.aborted, true);
    });
  });

  for (const wire of ['{"jsonrpc":"2.0","id":1e400,"result":{}}', '{"jsonrpc":"2.0","id":3,"error":{"code":1e400,"message":"overflow"}}']) {
    test(`${adapter} HTTP rejects nonfinite envelope numbers ${wire}`, async () => {
      await withClient(adapter, fixtureFetch(request => {
        if (request.method === "notifications/initialized") return new Response(null, { status: 202 });
        if (request.method !== "tools/call") return jsonResponse(request);
        return new Response(wire, { headers: { "Content-Type": "application/json" } });
      }), async call => { await assert.rejects(call(), /invalid.*JSON-RPC/i); });
    });
  }
}

test("HTTP response reading observes an already-aborted reader's rejection", async () => {
  const controller = new AbortController();
  controller.abort();
  const response = new Response(new ReadableStream<Uint8Array>({
    start(stream) { stream.error(new Error("body failed before reading")); },
  }));
  await assert.rejects(readMcpHttpJsonRpcResponse(response, 1, controller.signal), /body aborted/i);
  await new Promise(resolve => setImmediate(resolve));
});
