import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GlmAcpAgent } from "../protocol/agent.js";
import { SessionStore, type PersistedSession } from "../protocol/session-store.js";
import { SessionMcpTools } from "../tools/session-mcp-client.js";
import type { GlmMessage, GlmStreamChunk } from "../llm/glm-client.js";

function connection() {
  return {
    signal: new AbortController().signal,
    async sessionUpdate(params: Record<string, unknown>) { void params; },
    async requestPermission() {
      return { outcome: { outcome: "selected", optionId: "allow" } };
    },
  };
}

function textGlm(onCall: (messages: GlmMessage[]) => void = () => {}) {
  return {
    async *streamChat(messages: GlmMessage[]): AsyncGenerator<GlmStreamChunk> {
      onCall(structuredClone(messages));
      yield { text: "done" };
      yield { done: true, stopReason: "stop" };
    },
  };
}

class RejectingStore extends SessionStore {
  reject: (session: PersistedSession) => boolean = () => false;
  override async save(session: PersistedSession): Promise<void> {
    if (this.reject(session)) throw new Error("fixture checkpoint failed");
    await super.save(session);
  }
}

test("a completed file write survives provider failure and immediate fresh-agent resume", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-durable-write-"));
  const store = new SessionStore(join(cwd, "sessions"));
  let calls = 0;
  const agent = new GlmAcpAgent(connection() as never, { sessionStore: store, glm: {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      calls++;
      if (calls === 2) throw new Error("fixture provider failed");
      yield { toolCall: { id: "written", name: "write_file", arguments: JSON.stringify({ path: "effect.txt", content: "done" }) } };
      yield { done: true, stopReason: "tool_calls" };
    },
  } });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  try {
    await agent.setSessionMode({ sessionId, modeId: "accept_edits" });
    await assert.rejects(agent.prompt({ sessionId, prompt: [{ type: "text", text: "write it" }] }), /provider failed/);
    assert.equal(await readFile(join(cwd, "effect.txt"), "utf8"), "done");
    let resumed: GlmMessage[] = [];
    const restarted = new GlmAcpAgent(connection() as never, { sessionStore: new SessionStore(join(cwd, "sessions")), glm: textGlm(messages => { resumed = messages; }) });
    await restarted.resumeSession({ sessionId, cwd, mcpServers: [] });
    await restarted.prompt({ sessionId, prompt: [{ type: "text", text: "inspect the previous write" }] });
    const batch = resumed.filter(message => message.role === "tool" || (message.role === "assistant" && message.tool_calls?.length));
    assert.equal(batch.length, 2);
    assert.equal(batch[0]?.role, "assistant");
    assert.equal(batch[1]?.role, "tool");
    assert.equal(batch[1]?.role === "tool" && batch[1].tool_call_id, "written");
    await restarted.shutdown("disconnect");
  } finally {
    await agent.shutdown("disconnect");
    await rm(cwd, { recursive: true, force: true });
  }
});

test("uncertain and unstarted outcomes reach disk before the tool error returns", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-durable-uncertain-"));
  const store = new SessionStore(join(cwd, "sessions"));
  const conn = connection();
  conn.sessionUpdate = async params => {
    const update = params["update"] as { toolCallId?: string; status?: string };
    if (update.toolCallId === "first" && ["completed", "failed"].includes(update.status ?? "")) {
      throw new Error("fixture notification failed after write");
    }
  };
  const agent = new GlmAcpAgent(conn as never, { sessionStore: store, glm: {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      for (const id of ["first", "second"]) yield { toolCall: { id, name: "write_file", arguments: JSON.stringify({ path: id, content: "written" }) } };
      yield { done: true, stopReason: "tool_calls" };
    },
  } });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  try {
    await agent.setSessionMode({ sessionId, modeId: "accept_edits" });
    await assert.rejects(agent.prompt({ sessionId, prompt: [{ type: "text", text: "write both" }] }), /outcome.*unknown/i);
    assert.equal(await readFile(join(cwd, "first"), "utf8"), "written");
    await assert.rejects(readFile(join(cwd, "second")), { code: "ENOENT" });
    const tools = store.load(sessionId)?.messages.filter(message => message.role === "tool");
    assert.equal(tools?.length, 2);
    assert.match(String(tools?.[0]?.content), /outcome.*unknown/i);
    assert.match(String(tools?.[1]?.content), /not started/i);
  } finally {
    await agent.shutdown("disconnect");
    await rm(cwd, { recursive: true, force: true });
  }
});

test("a failed settled-tool checkpoint stops continuation and retains the live result", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-durable-tool-failure-"));
  const store = new RejectingStore(join(cwd, "sessions"));
  let calls = 0;
  let continued: GlmMessage[] = [];
  const agent = new GlmAcpAgent(connection() as never, { sessionStore: store, glm: {
    async *streamChat(messages: GlmMessage[]): AsyncGenerator<GlmStreamChunk> {
      calls++;
      if (calls === 1) {
        yield { toolCall: { id: "settled", name: "write_file", arguments: JSON.stringify({ path: "effect.txt", content: "done" }) } };
        yield { done: true, stopReason: "tool_calls" };
      } else {
        continued = structuredClone(messages);
        yield { text: "done" };
        yield { done: true, stopReason: "stop" };
      }
    },
  } });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  try {
    await agent.setSessionMode({ sessionId, modeId: "accept_edits" });
    store.reject = () => true;
    await assert.rejects(agent.prompt({ sessionId, prompt: [{ type: "text", text: "write it" }] }), /checkpoint failed/);
    assert.equal(calls, 1, "the provider must not continue before the checkpoint succeeds");
    store.reject = () => false;
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "inspect" }] });
    assert.ok(continued.some(message => message.role === "tool" && message.tool_call_id === "settled"));
  } finally {
    store.reject = () => false;
    await agent.shutdown("disconnect");
    await rm(cwd, { recursive: true, force: true });
  }
});

test("ordinary text prompt saves remain best effort", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-optional-save-"));
  const store = new RejectingStore(join(cwd, "sessions"));
  store.reject = () => true;
  const agent = new GlmAcpAgent(connection() as never, { sessionStore: store, glm: textGlm() });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  try {
    assert.equal((await agent.prompt({ sessionId, prompt: [{ type: "text", text: "hello" }] })).stopReason, "end_turn");
  } finally {
    store.reject = () => false;
    await agent.shutdown("disconnect");
    await rm(cwd, { recursive: true, force: true });
  }
});

test("fork parent checkpoint failure leaves its history usable without creating child resources", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-fork-save-failure-"));
  const store = new RejectingStore(join(cwd, "sessions"));
  let connections = 0;
  const agent = new GlmAcpAgent(connection() as never, { sessionStore: store, glm: textGlm(), connectSessionMcpServers: async () => { connections++; return new SessionMcpTools([]); } });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  try {
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "parent history" }] });
    store.reject = () => true;
    await assert.rejects(agent.unstable_forkSession({ sessionId, cwd, mcpServers: [] }), /checkpoint failed/);
    assert.equal(connections, 1);
    store.reject = () => false;
    assert.equal((await agent.prompt({ sessionId, prompt: [{ type: "text", text: "continue" }] })).stopReason, "end_turn");
  } finally {
    store.reject = () => false;
    await agent.shutdown("disconnect");
    await rm(cwd, { recursive: true, force: true });
  }
});

test("fork child checkpoint failure disposes provisional resources and leaves no orphaned child", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-fork-child-save-failure-"));
  const store = new RejectingStore(join(cwd, "sessions"));
  let connections = 0;
  let childDisposals = 0;
  const childTools = new SessionMcpTools([]);
  childTools.dispose = async () => { childDisposals++; };
  const agent = new GlmAcpAgent(connection() as never, { sessionStore: store, connectSessionMcpServers: async () => ++connections === 1 ? new SessionMcpTools([]) : childTools });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  try {
    store.reject = session => session.sessionId !== sessionId;
    await assert.rejects(agent.unstable_forkSession({ sessionId, cwd, mcpServers: [] }), /checkpoint failed/);
    assert.equal(childDisposals, 1);
    assert.equal((agent as unknown as { sessions: Map<string, unknown> }).sessions.size, 1);
    assert.equal((agent as unknown as { transitions: Map<string, unknown> }).transitions.size, 1);
  } finally {
    store.reject = () => false;
    await agent.shutdown("disconnect");
    await rm(cwd, { recursive: true, force: true });
  }
});

test("restore checkpoint failure retains the original resources and disposes the provisional replacement", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-restore-save-failure-"));
  const store = new RejectingStore(join(cwd, "sessions"));
  let connections = 0;
  let originalDisposals = 0;
  let replacementDisposals = 0;
  const original = new SessionMcpTools([]);
  original.dispose = async () => { originalDisposals++; };
  const replacement = new SessionMcpTools([]);
  replacement.dispose = async () => { replacementDisposals++; };
  let continued: GlmMessage[] = [];
  const agent = new GlmAcpAgent(connection() as never, { sessionStore: store, glm: textGlm(messages => { continued = messages; }), connectSessionMcpServers: async () => ++connections === 1 ? original : replacement });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  try {
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "retain history" }] });
    store.reject = () => true;
    await assert.rejects(agent.resumeSession({ sessionId, cwd, mcpServers: [] }), /checkpoint failed/);
    assert.equal(originalDisposals, 0);
    assert.equal(replacementDisposals, 1);
    store.reject = () => false;
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "continue" }] });
    assert.ok(continued.some(message => message.role === "user" && message.content === "retain history"));
  } finally {
    store.reject = () => false;
    await agent.shutdown("disconnect");
    await rm(cwd, { recursive: true, force: true });
  }
});

test("configuration changed during a restore checkpoint survives in the replacement and on disk", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-restore-config-save-"));
  const replacementCwd = join(cwd, "replacement");
  let saveStarted!: () => void;
  const saveReady = new Promise<void>(resolve => { saveStarted = resolve; });
  let releaseSave!: () => void;
  const saveHeld = new Promise<void>(resolve => { releaseSave = resolve; });
  class HeldStore extends SessionStore {
    holdNext = false;
    override async save(session: PersistedSession): Promise<void> {
      const hold = this.holdNext;
      this.holdNext = false;
      await super.save(session);
      if (hold) {
        saveStarted();
        await saveHeld;
      }
    }
  }
  const store = new HeldStore(join(cwd, "sessions"));
  const agent = new GlmAcpAgent(connection() as never, { sessionStore: store });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  try {
    store.holdNext = true;
    const restoring = agent.resumeSession({ sessionId, cwd: replacementCwd, mcpServers: [] });
    await saveReady;
    await agent.setSessionMode({ sessionId, modeId: "bypass_permissions" });
    releaseSave();
    assert.equal((await restoring).modes?.currentModeId, "bypass_permissions");
    assert.equal(store.load(sessionId)?.mode, "bypass_permissions");
    assert.equal(store.load(sessionId)?.cwd, replacementCwd);
  } finally {
    releaseSave();
    await agent.shutdown("disconnect");
    await rm(cwd, { recursive: true, force: true });
  }
});

test("failed close retains its live state and resources until a successful retry", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-close-save-failure-"));
  const store = new RejectingStore(join(cwd, "sessions"));
  let disposals = 0;
  const tools = new SessionMcpTools([]);
  tools.dispose = async () => { disposals++; };
  const agent = new GlmAcpAgent(connection() as never, { sessionStore: store, glm: textGlm(), connectSessionMcpServers: async () => tools });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  try {
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "retain me" }] });
    store.reject = () => true;
    await assert.rejects(agent.closeSession({ sessionId }), /checkpoint failed/);
    assert.equal(disposals, 0);
    assert.equal((agent as unknown as { sessions: Map<string, unknown> }).sessions.has(sessionId), true);
    store.reject = () => false;
    await agent.closeSession({ sessionId });
    assert.equal(disposals, 1);
    assert.ok(store.load(sessionId)?.messages.some(message => message.role === "user" && message.content === "retain me"));
    await assert.rejects(agent.prompt({ sessionId, prompt: [{ type: "text", text: "late" }] }), /session not found/i);
  } finally {
    store.reject = () => false;
    await agent.shutdown("disconnect");
    await rm(cwd, { recursive: true, force: true });
  }
});

test("an unloaded restore checkpoint failure leaves no installed state or provisional resources", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-unloaded-restore-save-failure-"));
  const store = new RejectingStore(join(cwd, "sessions"));
  const creator = new GlmAcpAgent(connection() as never, { sessionStore: store });
  const { sessionId } = await creator.newSession({ cwd, mcpServers: [] });
  await creator.closeSession({ sessionId });
  let disposals = 0;
  const tools = new SessionMcpTools([]);
  tools.dispose = async () => { disposals++; };
  const agent = new GlmAcpAgent(connection() as never, { sessionStore: store, connectSessionMcpServers: async () => tools });
  store.reject = () => true;
  try {
    await assert.rejects(agent.resumeSession({ sessionId, cwd, mcpServers: [] }), /checkpoint failed/);
    assert.equal(disposals, 1);
    assert.equal((agent as unknown as { sessions: Map<string, unknown> }).sessions.size, 0);
    assert.equal((agent as unknown as { transitions: Map<string, unknown> }).transitions.size, 0);
    assert.ok(store.load(sessionId), "the previous checkpoint remains recoverable");
  } finally {
    store.reject = () => false;
    await agent.shutdown("disconnect");
    await rm(cwd, { recursive: true, force: true });
  }
});

test("a deferred restore checkpoint failure reaches the draining prompt and releases its transition", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-deferred-save-failure-"));
  const store = new RejectingStore(join(cwd, "sessions"));
  let started!: () => void;
  const ready = new Promise<void>(resolve => { started = resolve; });
  let release!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  let calls = 0;
  const agent = new GlmAcpAgent(connection() as never, { sessionStore: store, sessionDrainTimeoutMs: 0, glm: {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      if (++calls === 1) {
        yield { text: "retained partial" };
        started();
        await held;
      }
      yield { done: true, stopReason: "stop" };
    },
  } });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "drain me" }] });
  try {
    await ready;
    store.reject = () => true;
    await assert.rejects(agent.resumeSession({ sessionId, cwd, mcpServers: [] }), /timed out/);
    const rejected = assert.rejects(prompt, /checkpoint failed/);
    release();
    await rejected;
    store.reject = () => false;
    assert.equal((await agent.prompt({ sessionId, prompt: [{ type: "text", text: "continue" }] })).stopReason, "end_turn");
  } finally {
    release();
    store.reject = () => false;
    await agent.shutdown("disconnect");
    await rm(cwd, { recursive: true, force: true });
  }
});

test("shutdown reports checkpoint failure while releasing all owned resources", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-shutdown-save-failure-"));
  const store = new RejectingStore(join(cwd, "sessions"));
  let disposals = 0;
  const tools = new SessionMcpTools([]);
  tools.dispose = async () => { disposals++; };
  const agent = new GlmAcpAgent(connection() as never, { sessionStore: store, connectSessionMcpServers: async () => tools });
  await agent.newSession({ cwd, mcpServers: [] });
  store.reject = () => true;
  try {
    await assert.rejects(agent.shutdown("disconnect"), /checkpoint failed/);
    assert.equal(disposals, 1);
    assert.equal((agent as unknown as { sessions: Map<string, unknown> }).sessions.size, 0);
  } finally {
    await rm(cwd, { recursive: true, force: true });
  }
});

test("a single failed shutdown checkpoint rejects with the original storage error", { timeout: 10_000 }, async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-shutdown-single-failure-"));
  const dir = join(cwd, "sessions");
  const agent = new GlmAcpAgent(connection() as never, { sessionStore: new SessionStore(dir) });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  // The checkpoint's rename destination becomes a directory, so the queued
  // atomic write rejects with the real errno error the store retains for flush.
  await mkdir(join(dir, `${sessionId}.json`), { recursive: true });
  try {
    await assert.rejects(agent.shutdown("disconnect"), (error: unknown) => {
      assert.ok(!(error instanceof AggregateError), "one storage failure must not be re-wrapped by flush reporting");
      assert.doesNotMatch(String(error), /cleanup and session persistence failed/);
      const errno = error as NodeJS.ErrnoException;
      // POSIX renames onto a directory as EISDIR; Windows as EPERM. What
      // matters is that the raw fs errno survives to the caller unwrapped.
      assert.ok(errno.code === "EISDIR" || errno.code === "EPERM", `unexpected errno ${errno.code}`);
      assert.match(errno.syscall ?? "", /rename/);
      return true;
    });
  } finally {
    await rm(cwd, { recursive: true, force: true });
  }
});

test("two distinct failed shutdown checkpoints still aggregate their raw errors once", { timeout: 10_000 }, async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-shutdown-two-failures-"));
  const dir = join(cwd, "sessions");
  const agent = new GlmAcpAgent(connection() as never, { sessionStore: new SessionStore(dir) });
  const first = await agent.newSession({ cwd, mcpServers: [] });
  const second = await agent.newSession({ cwd, mcpServers: [] });
  await mkdir(join(dir, `${first.sessionId}.json`), { recursive: true });
  await mkdir(join(dir, `${second.sessionId}.json`), { recursive: true });
  try {
    await assert.rejects(agent.shutdown("disconnect"), (error: unknown) => {
      assert.ok(error instanceof AggregateError);
      assert.match(error.message, /cleanup and session persistence failed/);
      assert.equal(error.errors.length, 2);
      for (const raw of error.errors as NodeJS.ErrnoException[]) {
        assert.ok(!(raw instanceof AggregateError), "each raw failure must surface unwrapped");
        assert.ok(raw.code === "EISDIR" || raw.code === "EPERM", `unexpected errno ${raw.code}`);
      }
      const reported = (error.errors as NodeJS.ErrnoException[]).map(raw => String(raw));
      assert.ok(reported.some(text => text.includes(first.sessionId)), "the first session's raw error is retained");
      assert.ok(reported.some(text => text.includes(second.sessionId)), "the second session's raw error is retained");
      return true;
    });
  } finally {
    await rm(cwd, { recursive: true, force: true });
  }
});
