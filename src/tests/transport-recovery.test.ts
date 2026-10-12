import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import OpenAI from "openai";
import { GlmAcpAgent } from "../protocol/agent.js";
import { SessionStore } from "../protocol/session-store.js";
import { PROVIDER_MAX_RETRIES, type GlmStreamChunk } from "../llm/glm-client.js";

test("a connection error cancels the turn and the next prompt continues the session", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-"));
  const store = new SessionStore(join(cwd, "sessions"));
  const notices: string[] = [];
  let calls = 0;
  const agent = new GlmAcpAgent({
    sessionUpdate: async (update: { update: { sessionUpdate: string; content?: { text?: string } } }) => {
      if (update.update.sessionUpdate === "agent_message_chunk" && update.update.content?.text) {
        notices.push(update.update.content.text);
      }
    },
  } as never, {
    sessionStore: store,
    visionClient: null,
    glm: {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        calls += 1;
        if (calls === 1) throw new OpenAI.APIConnectionError({ message: "Connection error." });
        yield { text: "still here" };
        yield { done: true, stopReason: "stop" };
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    const failed = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "add the missing row" }],
    });
    assert.equal(failed.stopReason, "cancelled");
    assert.match(notices.join("\n"), /Send the message again when you are back online/);
    const saved = store.load(sessionId);
    assert.equal(
      saved?.messages.some((message) => message.role === "user" && JSON.stringify(message).includes("add the missing row")),
      false
    );

    const continued = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "add the missing row" }],
    });
    assert.equal(continued.stopReason, "end_turn");
    assert.equal(notices.at(-1), "still here");
  } finally {
    await agent.closeSession({ sessionId });
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("partial text from a dropped stream is kept and the session accepts another prompt", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-partial-"));
  const store = new SessionStore(join(cwd, "sessions"));
  const notices: string[] = [];
  let calls = 0;
  const agent = new GlmAcpAgent({
    sessionUpdate: async (update: { update: { sessionUpdate: string; content?: { text?: string } } }) => {
      if (update.update.sessionUpdate === "agent_message_chunk" && update.update.content?.text) {
        notices.push(update.update.content.text);
      }
    },
  } as never, {
    sessionStore: store,
    visionClient: null,
    glm: {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        calls += 1;
        if (calls === 1) {
          yield { text: "partial answer" };
          throw Object.assign(new Error("socket hang up"), { code: "ECONNRESET" });
        }
        yield { text: "continued" };
        yield { done: true, stopReason: "stop" };
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    const failed = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "compare them" }],
    });
    assert.equal(failed.stopReason, "cancelled");
    assert.match(notices.join("\n"), /Send a message to continue/);
    const saved = store.load(sessionId);
    assert.equal(
      saved?.messages.some((message) => message.role === "assistant" && message.content === "partial answer"),
      true
    );

    const continued = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "continue" }],
    });
    assert.equal(continued.stopReason, "end_turn");
  } finally {
    await agent.closeSession({ sessionId });
    rmSync(cwd, { recursive: true, force: true });
  }
});

// Larger than the default model's compaction threshold, so the next prompt
// rewrites the live user message before the provider call.
const PRIOR_TURN = "prior-turn ".repeat(200_000);

test("a connection error after compaction drops the unsent prompt", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-compact-"));
  const store = new SessionStore(join(cwd, "sessions"));
  const notices: string[] = [];
  let sawCompaction = false;
  let seeded = false;
  const agent = new GlmAcpAgent({
    sessionUpdate: async (update: { update: { sessionUpdate: string; content?: { text?: string } } }) => {
      const text = update.update.content?.text;
      if (update.update.sessionUpdate === "agent_message_chunk" && text && !text.startsWith("prior-turn")) {
        notices.push(text);
      }
    },
  } as never, {
    sessionStore: store,
    visionClient: null,
    glm: {
      async *streamChat(messages: { role?: string; content?: unknown }[]): AsyncGenerator<GlmStreamChunk> {
        if (!seeded) {
          seeded = true;
          yield { text: PRIOR_TURN };
          yield { done: true, stopReason: "stop" };
          return;
        }
        sawCompaction = messages.some((message) => message.role === "user" && String(message.content).includes("Context compaction"));
        throw new OpenAI.APIConnectionError({ message: "Connection error." });
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    const prior = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "remember the bulk" }],
    });
    assert.equal(prior.stopReason, "end_turn");
    const failed = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "add the missing row" }],
    });
    assert.equal(sawCompaction, true);
    assert.equal(failed.stopReason, "cancelled");
    assert.match(notices.join("\n"), /Send the message again when you are back online/);
    const saved = store.load(sessionId);
    assert.equal(
      saved?.messages.some((message) => JSON.stringify(message).includes("add the missing row")),
      false
    );
    assert.equal(
      saved?.messages.some((message) => message.role === "user" && JSON.stringify(message).includes("remember the bulk")),
      true,
      "the prior user turn must survive a no-output drop after compaction"
    );
    assert.equal(
      saved?.messages.some((message) => message.role === "assistant" && String(message.content).startsWith("prior-turn")),
      true,
      "the prior assistant turn must survive a no-output drop after compaction"
    );
    await agent.closeSession({ sessionId });
    const afterClose = store.load(sessionId);
    assert.equal(
      afterClose?.messages.some((message) => message.role === "user" && JSON.stringify(message).includes("remember the bulk")),
      true,
      "closing the session must not replace the checkpoint with the shortened transcript"
    );
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("a queued prompt during a no-output drop after compaction keeps the prior exchange", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-compact-queued-"));
  const store = new SessionStore(join(cwd, "sessions"));
  let calls = 0;
  let sawCompaction = false;
  let release = (): void => {};
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  let markEntered = (): void => {};
  const entered = new Promise<void>((resolve) => {
    markEntered = resolve;
  });
  const agent = new GlmAcpAgent({
    sessionUpdate: async () => {},
  } as never, {
    sessionStore: store,
    visionClient: null,
    glm: {
      async *streamChat(messages: { role?: string; content?: unknown }[]): AsyncGenerator<GlmStreamChunk> {
        calls += 1;
        if (calls === 1) {
          yield { text: PRIOR_TURN };
          yield { done: true, stopReason: "stop" };
          return;
        }
        if (calls === 2) {
          sawCompaction = messages.some((message) => message.role === "user" && String(message.content).includes("Context compaction"));
          markEntered();
          await gate;
          throw Object.assign(new Error("socket hang up"), { code: "ECONNRESET" });
        }
        throw Object.assign(new Error("socket hang up"), { code: "ECONNRESET" });
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    const prior = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "remember the bulk" }],
    });
    assert.equal(prior.stopReason, "end_turn");
    const failing = agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "add the missing row" }],
    });
    await entered;
    const queued = agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "queued follow up" }],
    });
    release();
    const failed = await failing;
    const queuedResult = await queued;
    assert.equal(sawCompaction, true);
    assert.equal(failed.stopReason, "cancelled");
    assert.equal(queuedResult.stopReason, "cancelled");
    const saved = store.load(sessionId);
    assert.equal(
      saved?.messages.some((message) => message.role === "user" && JSON.stringify(message).includes("remember the bulk")),
      true
    );
    assert.equal(
      saved?.messages.some((message) => message.role === "assistant" && String(message.content).startsWith("prior-turn")),
      true
    );
    assert.equal(
      saved?.messages.some((message) => JSON.stringify(message).includes("add the missing row")),
      true,
      "a queued cancel keeps the prompt, the same as a cancel that did not compact"
    );
    assert.equal(saved?.messages.some((message) => JSON.stringify(message).includes("queued follow up")), false);
    await agent.closeSession({ sessionId });
    const afterClose = store.load(sessionId);
    assert.equal(
      afterClose?.messages.some((message) => message.role === "assistant" && String(message.content).startsWith("prior-turn")),
      true,
      "closing the session must not replace the checkpoint with the shortened transcript"
    );
  } finally {
    release();
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("a dropped stream after compaction keeps partial work", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-compact-partial-"));
  const store = new SessionStore(join(cwd, "sessions"));
  const notices: string[] = [];
  let sawCompaction = false;
  let seeded = false;
  const agent = new GlmAcpAgent({
    sessionUpdate: async (update: { update: { sessionUpdate: string; content?: { text?: string } } }) => {
      const text = update.update.content?.text;
      if (update.update.sessionUpdate === "agent_message_chunk" && text && !text.startsWith("prior-turn")) {
        notices.push(text);
      }
    },
  } as never, {
    sessionStore: store,
    visionClient: null,
    glm: {
      async *streamChat(messages: { role?: string; content?: unknown }[]): AsyncGenerator<GlmStreamChunk> {
        if (!seeded) {
          seeded = true;
          yield { text: PRIOR_TURN };
          yield { done: true, stopReason: "stop" };
          return;
        }
        sawCompaction = messages.some((message) => message.role === "user" && String(message.content).includes("Context compaction"));
        yield { text: "partial answer" };
        throw Object.assign(new Error("socket hang up"), { code: "ECONNRESET" });
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    const prior = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "remember the bulk" }],
    });
    assert.equal(prior.stopReason, "end_turn");
    const failed = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "compare them" }],
    });
    assert.equal(sawCompaction, true);
    assert.equal(failed.stopReason, "cancelled");
    assert.match(notices.join("\n"), /Send a message to continue/);
    const saved = store.load(sessionId);
    assert.equal(
      saved?.messages.some((message) => message.role === "assistant" && message.content === "partial answer"),
      true
    );
    assert.equal(
      saved?.messages.some((message) => message.role === "user" && JSON.stringify(message).includes("compare them")),
      true
    );
  } finally {
    await agent.closeSession({ sessionId });
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("every failed open cancels the turn", async () => {
  assert.equal(PROVIDER_MAX_RETRIES, 2);
  let attempts = 0;
  const server = createServer((req) => {
    attempts += 1;
    req.socket.destroy();
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("test server has no port");
  const savedKey = process.env["Z_AI_API_KEY"];
  const savedBase = process.env["ACP_GLM_BASE_URL"];
  process.env["Z_AI_API_KEY"] = "test-key";
  process.env["ACP_GLM_BASE_URL"] = `http://127.0.0.1:${address.port}/v1`;
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-exhaust-"));
  const store = new SessionStore(join(cwd, "sessions"));
  const agent = new GlmAcpAgent({ sessionUpdate: async () => {} } as never, {
    sessionStore: store,
    visionClient: null,
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    const failed = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "add the missing row" }],
    });
    assert.equal(failed.stopReason, "cancelled");
    assert.equal(attempts, PROVIDER_MAX_RETRIES + 1);
    const saved = store.load(sessionId);
    assert.equal(
      saved?.messages.some((message) => JSON.stringify(message).includes("add the missing row")),
      false
    );
  } finally {
    if (savedKey === undefined) delete process.env["Z_AI_API_KEY"];
    else process.env["Z_AI_API_KEY"] = savedKey;
    if (savedBase === undefined) delete process.env["ACP_GLM_BASE_URL"];
    else process.env["ACP_GLM_BASE_URL"] = savedBase;
    const dispatcher = (agent as unknown as { _glm?: { dispatcher?: { destroy: () => Promise<void> } } })._glm?.dispatcher;
    await dispatcher?.destroy().catch(() => undefined);
    await agent.shutdown("sigterm").catch(() => undefined);
    await new Promise<void>((resolve, reject) => server.close((err) => err ? reject(err) : resolve()));
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("replacing the shared pool lets another session finish the response it is reading", { timeout: 20_000 }, async () => {
  // One GlmClient serves every session. Retiring its pool with destroy()
  // aborts the other session's body (`TypeError: terminated`) and that prompt
  // rejects without a durable save. close() has to leave the read running.
  let releaseHealthy = (): void => {};
  const healthyHeld = new Promise<void>((resolve) => {
    releaseHealthy = () => resolve();
  });
  let markStreaming = (): void => {};
  const healthyStreaming = new Promise<void>((resolve) => {
    markStreaming = () => resolve();
  });
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("error", () => undefined);
    res.on("error", () => undefined);
    req.socket.on("error", () => undefined);
    req.on("end", () => {
      const body = Buffer.concat(chunks).toString("utf8");
      if (body.includes("drop the link")) {
        req.socket.destroy();
        return;
      }
      if (!body.includes("hold this stream")) {
        res.writeHead(500, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: "unexpected request" } }));
        return;
      }
      res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
      res.socket?.setNoDelay(true);
      res.write(`data: ${JSON.stringify({
        id: "chatcmpl-healthy",
        object: "chat.completion.chunk",
        choices: [{ index: 0, delta: { content: "kept alive" }, finish_reason: null }],
      })}\n\n`);
      void healthyHeld.then(() => {
        if (res.writableEnded) return;
        res.write(`data: ${JSON.stringify({
          id: "chatcmpl-healthy",
          object: "chat.completion.chunk",
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
        })}\n\n`);
        res.end("data: [DONE]\n\n");
      });
    });
  });
  server.on("clientError", () => undefined);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("test server has no port");
  const savedKey = process.env["Z_AI_API_KEY"];
  const savedBase = process.env["ACP_GLM_BASE_URL"];
  process.env["Z_AI_API_KEY"] = "test-key";
  process.env["ACP_GLM_BASE_URL"] = `http://127.0.0.1:${address.port}/v1`;
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-shared-"));
  const store = new SessionStore(join(cwd, "sessions"));
  const agent = new GlmAcpAgent({
    sessionUpdate: async (update: { update: { sessionUpdate: string; content?: { text?: string } } }) => {
      if (update.update.content?.text?.includes("kept alive")) markStreaming();
    },
  } as never, {
    sessionStore: store,
    visionClient: null,
  });
  const healthyId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  const droppedId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  const poolOf = (): { closed: boolean; destroyed: boolean; destroy: () => Promise<void> } => {
    const dispatcher = (agent as unknown as {
      _glm?: { dispatcher?: { closed: boolean; destroyed: boolean; destroy: () => Promise<void> } };
    })._glm?.dispatcher;
    if (!dispatcher) throw new Error("shared provider client was not created");
    return dispatcher;
  };
  const healthy = agent.prompt({
    sessionId: healthyId,
    prompt: [{ type: "text", text: "hold this stream" }],
  });
  try {
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("healthy session did not start streaming")), 5_000);
      healthyStreaming.then(() => {
        clearTimeout(timer);
        resolve();
      }, (err: unknown) => {
        clearTimeout(timer);
        reject(err);
      });
    });
    const previous = poolOf();
    const failed = await agent.prompt({
      sessionId: droppedId,
      prompt: [{ type: "text", text: "drop the link" }],
    });
    assert.equal(failed.stopReason, "cancelled");
    assert.notEqual(poolOf(), previous);
    // close() marks the pool closed immediately and destroys it only after
    // the in-flight body ends. destroy() would already report destroyed.
    assert.equal(previous.closed, true);
    assert.equal(previous.destroyed, false);
    releaseHealthy();
    const continued = await healthy;
    assert.equal(continued.stopReason, "end_turn");
    const saved = store.load(healthyId);
    assert.equal(
      saved?.messages.some((message) => message.role === "assistant" && message.content === "kept alive"),
      true
    );
    const dropped = store.load(droppedId);
    assert.equal(
      dropped?.messages.some((message) => JSON.stringify(message).includes("drop the link")),
      false
    );
  } finally {
    releaseHealthy();
    await healthy.catch(() => undefined);
    if (savedKey === undefined) delete process.env["Z_AI_API_KEY"];
    else process.env["Z_AI_API_KEY"] = savedKey;
    if (savedBase === undefined) delete process.env["ACP_GLM_BASE_URL"];
    else process.env["ACP_GLM_BASE_URL"] = savedBase;
    const dispatcher = (agent as unknown as {
      _glm?: { dispatcher?: { destroy: () => Promise<void> } };
    })._glm?.dispatcher;
    await dispatcher?.destroy().catch(() => undefined);
    await agent.shutdown("sigterm").catch(() => undefined);
    await new Promise<void>((resolve, reject) => server.close((err) => err ? reject(err) : resolve()));
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("a session waiting to retry uses the replacement pool", { timeout: 20_000 }, async () => {
  // Session B is inside the SDK's retry sleep when session A drops and
  // retires the shared pool. B's next attempt has to reach the provider on
  // the replacement; a dispatch on the closed pool becomes APIConnectionError
  // and the unanswered prompt is removed.
  const retryAfterMs = 4_000;
  let markBackoff = (): void => {};
  const backoffStarted = new Promise<void>((resolve) => {
    markBackoff = () => resolve();
  });
  let retryAttempts = 0;
  let dropAttempts = 0;
  let poolRebuilt = false;
  let sawRetryAfterRebuild = false;
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("error", () => undefined);
    res.on("error", () => undefined);
    req.socket.on("error", () => undefined);
    req.on("end", () => {
      const body = Buffer.concat(chunks).toString("utf8");
      if (body.includes("drop the link")) {
        dropAttempts += 1;
        req.socket.destroy();
        return;
      }
      if (!body.includes("wait out the limit")) {
        res.writeHead(500, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: "unexpected request" } }));
        return;
      }
      retryAttempts += 1;
      if (retryAttempts === 1) {
        res.writeHead(429, {
          "content-type": "application/json",
          "retry-after-ms": String(retryAfterMs),
        });
        res.end(JSON.stringify({ error: { message: "slow down", type: "server_error" } }));
        markBackoff();
        return;
      }
      if (poolRebuilt) sawRetryAfterRebuild = true;
      res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
      res.end(
        `data: ${JSON.stringify({
          id: "chatcmpl-retry",
          object: "chat.completion.chunk",
          choices: [{ index: 0, delta: { content: "retry landed" }, finish_reason: null }],
        })}\n\n` +
        `data: ${JSON.stringify({
          id: "chatcmpl-retry",
          object: "chat.completion.chunk",
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
        })}\n\n` +
        "data: [DONE]\n\n"
      );
    });
  });
  server.on("clientError", () => undefined);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("test server has no port");
  const savedKey = process.env["Z_AI_API_KEY"];
  const savedBase = process.env["ACP_GLM_BASE_URL"];
  process.env["Z_AI_API_KEY"] = "test-key";
  process.env["ACP_GLM_BASE_URL"] = `http://127.0.0.1:${address.port}/v1`;
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-retry-"));
  const store = new SessionStore(join(cwd, "sessions"));
  const agent = new GlmAcpAgent({ sessionUpdate: async () => {} } as never, {
    sessionStore: store,
    visionClient: null,
  });
  const retryId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  const droppedId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  const poolOf = (): { closed: boolean; destroyed: boolean } => {
    const dispatcher = (agent as unknown as {
      _glm?: { dispatcher?: { closed: boolean; destroyed: boolean } };
    })._glm?.dispatcher;
    if (!dispatcher) throw new Error("shared provider client was not created");
    return dispatcher;
  };
  const retrying = agent.prompt({
    sessionId: retryId,
    prompt: [{ type: "text", text: "wait out the limit" }],
  });
  try {
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("retry backoff did not start")), 5_000);
      backoffStarted.then(() => {
        clearTimeout(timer);
        resolve();
      }, (err: unknown) => {
        clearTimeout(timer);
        reject(err);
      });
    });
    const previous = poolOf();
    const failed = await agent.prompt({
      sessionId: droppedId,
      prompt: [{ type: "text", text: "drop the link" }],
    });
    poolRebuilt = true;
    assert.equal(failed.stopReason, "cancelled");
    assert.equal(dropAttempts, PROVIDER_MAX_RETRIES + 1);
    assert.notEqual(poolOf(), previous);
    // The 429 response has already finished, so this pool is idle and closed
    // while B is still asleep. B's next attempt must not dispatch on it.
    assert.equal(previous.closed, true);
    const continued = await retrying;
    assert.equal(continued.stopReason, "end_turn");
    assert.equal(retryAttempts, 2);
    assert.equal(sawRetryAfterRebuild, true);
    const saved = store.load(retryId);
    assert.equal(
      saved?.messages.some((message) => message.role === "assistant" && message.content === "retry landed"),
      true
    );
    assert.equal(
      saved?.messages.some((message) => message.role === "user" && JSON.stringify(message).includes("wait out the limit")),
      true
    );
    const dropped = store.load(droppedId);
    assert.equal(
      dropped?.messages.some((message) => JSON.stringify(message).includes("drop the link")),
      false
    );
  } finally {
    await retrying.catch(() => undefined);
    if (savedKey === undefined) delete process.env["Z_AI_API_KEY"];
    else process.env["Z_AI_API_KEY"] = savedKey;
    if (savedBase === undefined) delete process.env["ACP_GLM_BASE_URL"];
    else process.env["ACP_GLM_BASE_URL"] = savedBase;
    const dispatcher = (agent as unknown as {
      _glm?: { dispatcher?: { destroy: () => Promise<void> } };
    })._glm?.dispatcher;
    await dispatcher?.destroy().catch(() => undefined);
    await agent.shutdown("sigterm").catch(() => undefined);
    await new Promise<void>((resolve, reject) => server.close((err) => err ? reject(err) : resolve()));
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("two connection drops in a row both cancel and the session stays usable", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-twice-"));
  const store = new SessionStore(join(cwd, "sessions"));
  const notices: string[] = [];
  let calls = 0;
  const agent = new GlmAcpAgent({
    sessionUpdate: async (update: { update: { sessionUpdate: string; content?: { text?: string } } }) => {
      if (update.update.sessionUpdate === "agent_message_chunk" && update.update.content?.text) {
        notices.push(update.update.content.text);
      }
    },
  } as never, {
    sessionStore: store,
    visionClient: null,
    glm: {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        calls += 1;
        if (calls < 3) throw new OpenAI.APIConnectionError({ message: "Connection error." });
        yield { text: "third time" };
        yield { done: true, stopReason: "stop" };
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    const first = await agent.prompt({ sessionId, prompt: [{ type: "text", text: "first attempt" }] });
    const second = await agent.prompt({ sessionId, prompt: [{ type: "text", text: "second attempt" }] });
    assert.equal(first.stopReason, "cancelled");
    assert.equal(second.stopReason, "cancelled");
    assert.match(notices.join("\n"), /Send the message again when you are back online/);
    const saved = store.load(sessionId);
    assert.equal(saved?.messages.some((message) => JSON.stringify(message).includes("first attempt")), false);
    assert.equal(saved?.messages.some((message) => JSON.stringify(message).includes("second attempt")), false);
    const third = await agent.prompt({ sessionId, prompt: [{ type: "text", text: "third attempt" }] });
    assert.equal(third.stopReason, "end_turn");
    assert.equal(notices.at(-1), "third time");
  } finally {
    await agent.closeSession({ sessionId });
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("a connection drop after a tool result keeps that work", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-tool-"));
  const store = new SessionStore(join(cwd, "sessions"));
  const notices: string[] = [];
  let calls = 0;
  const agent = new GlmAcpAgent({
    sessionUpdate: async (update: { update: { sessionUpdate: string; content?: { text?: string } } }) => {
      if (update.update.sessionUpdate === "agent_message_chunk" && update.update.content?.text) {
        notices.push(update.update.content.text);
      }
    },
  } as never, {
    sessionStore: store,
    visionClient: null,
    glm: {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        calls += 1;
        if (calls === 1) {
          yield {
            toolCall: {
              id: "list-1",
              name: "list_files",
              arguments: JSON.stringify({ path: "." }),
            },
          };
          yield { done: true, stopReason: "tool_calls" };
          return;
        }
        throw new OpenAI.APIConnectionError({ message: "Connection error." });
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    const failed = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "list the directory" }],
    });
    assert.equal(failed.stopReason, "cancelled");
    const notice = notices.join("\n");
    assert.match(notice, /Send a message to continue/);
    assert.doesNotMatch(notice, /Send the message again/);
    const saved = store.load(sessionId);
    assert.equal(
      saved?.messages.some((message) => message.role === "user" && JSON.stringify(message).includes("list the directory")),
      true
    );
    assert.equal(saved?.messages.some((message) => message.role === "tool" && message.tool_call_id === "list-1"), true);
    assert.equal(
      saved?.messages.some((message) => message.role === "assistant" && typeof message.content === "string" && message.content.length > 0),
      false
    );
  } finally {
    await agent.closeSession({ sessionId });
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("cancelling keeps the prompt with and without compaction", async () => {
  const plainCwd = mkdtempSync(join(tmpdir(), "glm-transport-cancel-plain-"));
  const plainStore = new SessionStore(join(plainCwd, "sessions"));
  let markPlain = (): void => {};
  const plainWaiting = new Promise<void>((resolve) => {
    markPlain = resolve;
  });
  const plainAgent = new GlmAcpAgent({ sessionUpdate: async () => {} } as never, {
    sessionStore: plainStore,
    visionClient: null,
    glm: {
      async *streamChat(_messages: unknown, signal?: AbortSignal): AsyncGenerator<GlmStreamChunk> {
        markPlain();
        await new Promise<void>((resolve) => {
          if (signal?.aborted) return resolve();
          signal?.addEventListener("abort", () => resolve(), { once: true });
        });
        if (signal?.aborted) return;
        yield { text: "should not run" };
        yield { done: true, stopReason: "stop" };
      },
    },
  });
  const plainId = (await plainAgent.newSession({ cwd: plainCwd, mcpServers: [] })).sessionId;
  const compactCwd = mkdtempSync(join(tmpdir(), "glm-transport-cancel-compact-"));
  const compactStore = new SessionStore(join(compactCwd, "sessions"));
  let calls = 0;
  let sawCompaction = false;
  let releaseCompact = (): void => {};
  const compactGate = new Promise<void>((resolve) => {
    releaseCompact = resolve;
  });
  let markEntered = (): void => {};
  const entered = new Promise<void>((resolve) => {
    markEntered = resolve;
  });
  const compactAgent = new GlmAcpAgent({ sessionUpdate: async () => {} } as never, {
    sessionStore: compactStore,
    visionClient: null,
    glm: {
      async *streamChat(messages: { role?: string; content?: unknown }[], signal?: AbortSignal): AsyncGenerator<GlmStreamChunk> {
        calls += 1;
        if (calls === 1) {
          yield { text: PRIOR_TURN };
          yield { done: true, stopReason: "stop" };
          return;
        }
        sawCompaction = messages.some((message) => message.role === "user" && String(message.content).includes("Context compaction"));
        markEntered();
        await compactGate;
        if (signal?.aborted) return;
        yield { text: "should not run" };
        yield { done: true, stopReason: "stop" };
      },
    },
  });
  const compactId = (await compactAgent.newSession({ cwd: compactCwd, mcpServers: [] })).sessionId;
  try {
    const plainPrompt = plainAgent.prompt({ sessionId: plainId, prompt: [{ type: "text", text: "keep this prompt" }] });
    await plainWaiting;
    await plainAgent.cancel({ sessionId: plainId });
    assert.equal((await plainPrompt).stopReason, "cancelled");
    const plainSaved = plainStore.load(plainId);
    assert.equal(
      plainSaved?.messages.some((message) => JSON.stringify(message).includes("keep this prompt")),
      true
    );

    const prior = await compactAgent.prompt({
      sessionId: compactId,
      prompt: [{ type: "text", text: "remember the bulk" }],
    });
    assert.equal(prior.stopReason, "end_turn");
    const compactPrompt = compactAgent.prompt({
      sessionId: compactId,
      prompt: [{ type: "text", text: "keep after compaction" }],
    });
    await entered;
    assert.equal(sawCompaction, true);
    await compactAgent.cancel({ sessionId: compactId });
    releaseCompact();
    assert.equal((await compactPrompt).stopReason, "cancelled");
    await compactAgent.shutdown("sigterm");
    const saved = compactStore.load(compactId);
    assert.equal(saved?.messages.some((message) => JSON.stringify(message).includes("keep after compaction")), true);
    assert.equal(
      saved?.messages.some((message) => message.role === "user" && JSON.stringify(message).includes("remember the bulk")),
      true
    );
    assert.equal(
      saved?.messages.some((message) => message.role === "assistant" && String(message.content).startsWith("prior-turn")),
      true
    );
  } finally {
    releaseCompact();
    await plainAgent.shutdown("sigterm").catch(() => undefined);
    await compactAgent.shutdown("sigterm").catch(() => undefined);
    rmSync(plainCwd, { recursive: true, force: true });
    rmSync(compactCwd, { recursive: true, force: true });
  }
});

test("shutdown after a compacted connection drop keeps the prior checkpoint", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-shutdown-"));
  const store = new SessionStore(join(cwd, "sessions"));
  let seeded = false;
  const agent = new GlmAcpAgent({ sessionUpdate: async () => {} } as never, {
    sessionStore: store,
    visionClient: null,
    glm: {
      async *streamChat(messages: { role?: string; content?: unknown }[]): AsyncGenerator<GlmStreamChunk> {
        if (!seeded) {
          seeded = true;
          yield { text: PRIOR_TURN };
          yield { done: true, stopReason: "stop" };
          return;
        }
        const compacted = messages.some((message) => message.role === "user" && String(message.content).includes("Context compaction"));
        if (!compacted) throw new Error("expected compaction before the connection drop");
        throw new OpenAI.APIConnectionError({ message: "Connection error." });
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    const prior = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "remember the bulk" }],
    });
    assert.equal(prior.stopReason, "end_turn");
    const failed = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "add the missing row" }],
    });
    assert.equal(failed.stopReason, "cancelled");
    await agent.shutdown("sigterm");
    const saved = store.load(sessionId);
    assert.equal(saved?.messages.some((message) => JSON.stringify(message).includes("add the missing row")), false);
    assert.equal(
      saved?.messages.some((message) => message.role === "user" && JSON.stringify(message).includes("remember the bulk")),
      true
    );
    assert.equal(
      saved?.messages.some((message) => message.role === "assistant" && String(message.content).startsWith("prior-turn")),
      true
    );
  } finally {
    await agent.shutdown("sigterm").catch(() => undefined);
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("a cancel during the recovery save skips the notice and still returns", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-recheck-"));
  const store = new SessionStore(join(cwd, "sessions"));
  let releaseSave = (): void => {};
  const saveGate = new Promise<void>((resolve) => {
    releaseSave = resolve;
  });
  let markSave = (): void => {};
  const saveStarted = new Promise<void>((resolve) => {
    markSave = resolve;
  });
  const originalSave = store.save.bind(store);
  let gated = false;
  store.save = (session) => {
    if (!gated) {
      gated = true;
      markSave();
      return saveGate.then(() => originalSave(session));
    }
    return originalSave(session);
  };
  let noticeStarted = false;
  const agent = new GlmAcpAgent({
    sessionUpdate: async (update: { update: { content?: { text?: string } } }) => {
      if (update.update.content?.text?.includes("Connection lost")) {
        noticeStarted = true;
        await new Promise<void>(() => {});
      }
    },
  } as never, {
    sessionStore: store,
    visionClient: null,
    glm: {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        yield* [];
        throw new OpenAI.APIConnectionError({ message: "Connection error." });
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "add the missing row" }] });
  let close: Promise<void> | undefined;
  try {
    await saveStarted;
    await agent.cancel({ sessionId });
    close = agent.closeSession({ sessionId });
    releaseSave();
    const settled = await Promise.race([
      Promise.all([prompt, close]).then(() => "settled" as const),
      new Promise<"timeout">((resolve) => setTimeout(() => resolve("timeout"), 2_000)),
    ]);
    assert.equal(settled, "settled");
    assert.equal((await prompt).stopReason, "cancelled");
    assert.equal(noticeStarted, false);
    const saved = store.load(sessionId);
    assert.equal(saved?.messages.some((message) => JSON.stringify(message).includes("add the missing row")), false);
  } finally {
    releaseSave();
    await Promise.allSettled([prompt, close]);
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("a held connection-lost notice does not block cancel or close", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-notice-"));
  const store = new SessionStore(join(cwd, "sessions"));
  let releaseNotice = (): void => {};
  const noticeGate = new Promise<void>((resolve) => {
    releaseNotice = resolve;
  });
  let markNotice = (): void => {};
  const noticeStarted = new Promise<void>((resolve) => {
    markNotice = resolve;
  });
  const agent = new GlmAcpAgent({
    sessionUpdate: async (update: { update: { content?: { text?: string } } }) => {
      if (update.update.content?.text?.includes("Connection lost")) {
        markNotice();
        await noticeGate;
      }
    },
  } as never, {
    sessionStore: store,
    visionClient: null,
    glm: {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        yield* [];
        throw new OpenAI.APIConnectionError({ message: "Connection error." });
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "add the missing row" }] });
  let close: Promise<void> | undefined;
  try {
    await noticeStarted;
    await agent.cancel({ sessionId });
    close = agent.closeSession({ sessionId });
    const settled = await Promise.race([
      Promise.all([prompt, close]).then(() => "settled" as const),
      new Promise<"timeout">((resolve) => setTimeout(() => resolve("timeout"), 2_000)),
    ]);
    assert.equal(settled, "settled");
    assert.equal((await prompt).stopReason, "cancelled");
  } finally {
    releaseNotice();
    await Promise.allSettled([prompt, close]);
    await agent.shutdown("sigterm").catch(() => undefined);
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("an HTTP provider error still fails the prompt", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-http-"));
  const agent = new GlmAcpAgent({ sessionUpdate: async () => {} } as never, {
    sessionStore: null,
    visionClient: null,
    glm: {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        const fail = { status: 502 };
        if (fail.status >= 500) throw Object.assign(new Error("upstream"), fail);
        yield { text: "unreachable" };
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    await assert.rejects(
      agent.prompt({ sessionId, prompt: [{ type: "text", text: "hello" }] }),
      /upstream/
    );
  } finally {
    await agent.closeSession({ sessionId });
    rmSync(cwd, { recursive: true, force: true });
  }
});
