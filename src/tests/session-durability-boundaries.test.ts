import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import { execFile } from "node:child_process";
import { syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { GlmAcpAgent } from "../protocol/agent.js";
import { SessionStore, type PersistedSession } from "../protocol/session-store.js";
import { SessionMcpTools } from "../tools/session-mcp-client.js";
import type { GlmStreamChunk } from "../llm/glm-client.js";

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>(done => { resolve = done; });
  return { promise, resolve };
}

function connection() {
  return {
    signal: new AbortController().signal,
    async sessionUpdate() {},
    async requestPermission() { return { outcome: { outcome: "selected", optionId: "allow" } }; },
  };
}

function textGlm() {
  return { async *streamChat(): AsyncGenerator<GlmStreamChunk> {
    yield { text: "done" };
    yield { done: true, stopReason: "stop" };
  } };
}

async function withCheckpointWrites(
  dir: string,
  onWritten: (snapshot: PersistedSession, path: string) => Promise<void>,
  run: () => Promise<void>,
) {
  const originalOpen = fs.open;
  fs.open = (async (path, ...args) => {
    const handle = await originalOpen(path, ...args);
    if (dirname(String(path)) === dir && basename(String(path)).startsWith(".")) {
      const write = handle.writeFile.bind(handle);
      handle.writeFile = async (data, ...writeArgs) => {
        await write(data, ...writeArgs);
        await onWritten(JSON.parse(Buffer.from(data as Uint8Array).toString("utf8")) as PersistedSession, String(path));
      };
    }
    return handle;
  }) as typeof fs.open;
  syncBuiltinESMExports();
  try { await run(); } finally {
    fs.open = originalOpen;
    syncBuiltinESMExports();
  }
}

for (const mode of ["close", "shutdown"] as const) {
  test(`a fork cancelled by ${mode} during child persistence leaves no durable child`, { timeout: 10_000 }, async () => {
    const cwd = await fs.mkdtemp(join(tmpdir(), "glm-fork-rollback-"));
    const dir = join(cwd, "sessions");
    const store = new SessionStore(dir);
    const written = deferred();
    const release = deferred();
    let parentId = "";
    let childId = "";
    let tempPath = "";
    let childSignal: AbortSignal | undefined;
    let connections = 0;
    const disposals = [0, 0];
    const tools = [new SessionMcpTools([]), new SessionMcpTools([])];
    for (const [index, owned] of tools.entries()) {
      const dispose = owned.dispose.bind(owned);
      owned.dispose = async () => { disposals[index]!++; await dispose(); };
    }
    const agent = new GlmAcpAgent(connection() as never, {
      sessionStore: store, glm: textGlm(),
      connectSessionMcpServers: async (_servers, signal) => {
        if (connections === 1) childSignal = signal;
        return tools[connections++]!;
      },
    });
    let fresh: GlmAcpAgent | undefined;
    let fork: Promise<unknown> | undefined;
    let stopping: Promise<void> | undefined;
    try {
      parentId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
      await agent.prompt({ sessionId: parentId, prompt: [{ type: "text", text: "parent history" }] });
      await withCheckpointWrites(dir, async (snapshot, path) => {
        if (snapshot.sessionId === parentId) return;
        childId = snapshot.sessionId;
        tempPath = path;
        written.resolve();
        await release.promise;
      }, async () => {
        fork = agent.unstable_forkSession({ sessionId: parentId, cwd, mcpServers: [] });
        const rejected = assert.rejects(fork, /Session fork cancelled/);
        await written.promise;
        await fs.access(tempPath);
        await assert.rejects(fs.access(join(dir, `${childId}.json`)), { code: "ENOENT" });
        stopping = mode === "close" ? agent.closeSession({ sessionId: parentId }) : agent.shutdown("disconnect");
        assert.equal(childSignal?.aborted, true);
        release.resolve();
        await rejected;
        await stopping;
        await assert.rejects(fs.access(join(dir, `${childId}.json`)), { code: "ENOENT" });
      });
      assert.deepEqual(disposals, [1, 1]);
      assert.deepEqual((await agent.listSessions({ cwd })).sessions.map(session => session.sessionId), [parentId]);
      const freshStore = new SessionStore(dir);
      assert.equal(await freshStore.loadAsync(childId), undefined);
      fresh = new GlmAcpAgent(connection() as never, { sessionStore: freshStore, glm: textGlm() });
      await assert.rejects(fresh.resumeSession({ sessionId: childId, cwd, mcpServers: [] }), /not found/i);
      await fresh.resumeSession({ sessionId: parentId, cwd, mcpServers: [] });
      assert.ok((await freshStore.loadAsync(parentId))?.messages.some(message => message.role === "user" && message.content === "parent history"));
      const retry = await fresh.unstable_forkSession({ sessionId: parentId, cwd, mcpServers: [] });
      assert.notEqual(retry.sessionId, childId);
      assert.deepEqual(new Set((await fresh.listSessions({ cwd })).sessions.map(session => session.sessionId)), new Set([parentId, retry.sessionId]));
    } finally {
      release.resolve();
      await Promise.allSettled([fork, stopping]);
      await fresh?.shutdown("disconnect");
      await agent.shutdown("disconnect");
      await fs.rm(cwd, { recursive: true, force: true });
    }
  });
}

test("cancelled fork rollback waits for every accepted child write", { timeout: 10_000 }, async () => {
  const cwd = await fs.mkdtemp(join(tmpdir(), "glm-fork-rollback-order-"));
  const dir = join(cwd, "sessions");
  const store = new SessionStore(dir);
  const firstWritten = deferred();
  const releaseFirst = deferred();
  const secondWritten = deferred();
  const releaseSecond = deferred();
  let parentId = "";
  let child: PersistedSession | undefined;
  let childWrites = 0;
  const agent = new GlmAcpAgent(connection() as never, { sessionStore: store, glm: textGlm() });
  let fork: Promise<unknown> | undefined;
  let stopping: Promise<void> | undefined;
  let queued: Promise<void> | undefined;
  try {
    parentId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
    await withCheckpointWrites(dir, async snapshot => {
      if (snapshot.sessionId === parentId) return;
      if (++childWrites === 1) {
        child = snapshot;
        firstWritten.resolve();
        await releaseFirst.promise;
      } else {
        secondWritten.resolve();
        await releaseSecond.promise;
      }
    }, async () => {
      let forkSettled = false;
      fork = agent.unstable_forkSession({ sessionId: parentId, cwd, mcpServers: [] });
      void fork.then(() => { forkSettled = true; }, () => { forkSettled = true; });
      const rejected = assert.rejects(fork, /Session fork cancelled/);
      await firstWritten.promise;
      queued = store.save({ ...child!, title: "accepted later checkpoint" });
      stopping = agent.closeSession({ sessionId: parentId });
      releaseFirst.resolve();
      await secondWritten.promise;
      await new Promise<void>(resolve => setImmediate(resolve));
      assert.equal(forkSettled, false, "fork rejection waits for the queued write before removing its record");
      releaseSecond.resolve();
      await Promise.all([queued, rejected, stopping]);
      await assert.rejects(fs.access(join(dir, `${child!.sessionId}.json`)), { code: "ENOENT" });
      assert.equal(await new SessionStore(dir).loadAsync(child!.sessionId), undefined);
    });
  } finally {
    releaseFirst.resolve();
    releaseSecond.resolve();
    await Promise.allSettled([fork, stopping, queued]);
    await agent.shutdown("disconnect");
    await fs.rm(cwd, { recursive: true, force: true });
  }
});

for (const disposalFails of [false, true]) {
  test(disposalFails
    ? "cancelled fork retains rollback failure when provisional MCP disposal also fails"
    : "cancelled fork reports a failed durable-child rollback to its caller", { timeout: 10_000 }, async () => {
    const cwd = await fs.mkdtemp(join(tmpdir(), "glm-fork-rollback-error-"));
    const dir = join(cwd, "sessions");
    const store = new SessionStore(dir);
    const written = deferred();
    const release = deferred();
    let parentId = "";
    let childId = "";
    const failure = Object.assign(new Error("fixture unlink denied"), { code: "EACCES" });
    const disposalFailure = new Error("fixture provisional disposal failed");
    let childDisposals = 0;
    let connections = 0;
    const childTools = new SessionMcpTools([]);
    const dispose = childTools.dispose.bind(childTools);
    childTools.dispose = async () => {
      childDisposals++;
      await dispose();
      if (disposalFails) throw disposalFailure;
    };
    const originalUnlink = fs.unlink;
    const agent = new GlmAcpAgent(connection() as never, {
      sessionStore: store, glm: textGlm(),
      connectSessionMcpServers: async () => ++connections === 1 ? new SessionMcpTools([]) : childTools,
    });
    let fork: Promise<unknown> | undefined;
    let stopping: Promise<void> | undefined;
    try {
      parentId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
      fs.unlink = async path => {
        if (String(path) === join(dir, `${childId}.json`)) throw failure;
        await originalUnlink(path);
      };
      syncBuiltinESMExports();
      await withCheckpointWrites(dir, async snapshot => {
        if (snapshot.sessionId === parentId) return;
        childId = snapshot.sessionId;
        written.resolve();
        await release.promise;
      }, async () => {
        fork = agent.unstable_forkSession({ sessionId: parentId, cwd, mcpServers: [] });
        const rejected = assert.rejects(fork, error => {
          assert.ok(error instanceof AggregateError);
          assert.match(error.message, /rollback failed/i);
          assert.match(error.message, new RegExp(childId));
          const rollback = disposalFails ? error.errors[0] : error;
          assert.ok(rollback instanceof AggregateError);
          assert.match(String(rollback.errors[0]), /Session fork cancelled/);
          assert.equal(rollback.errors[1], failure);
          assert.equal(rollback.cause, failure);
          if (disposalFails) {
            assert.equal(error.errors[1], disposalFailure);
            assert.equal(error.cause, disposalFailure);
          }
          return true;
        });
        await written.promise;
        stopping = agent.closeSession({ sessionId: parentId });
        release.resolve();
        await rejected;
        await stopping;
        assert.equal(childDisposals, 1);
        await assert.rejects(agent.prompt({ sessionId: parentId, prompt: [{ type: "text", text: "after close" }] }), /session not found/i);
        assert.ok(await new SessionStore(dir).loadAsync(childId), "failed removal remains recoverable and its error is visible");
      });
      await assert.rejects(store.flush(), error => error instanceof AggregateError && error.errors.includes(failure));
    } finally {
      release.resolve();
      await Promise.allSettled([fork, stopping]);
      fs.unlink = originalUnlink;
      syncBuiltinESMExports();
      await store.flush().catch(() => undefined);
      await agent.shutdown("disconnect");
      await fs.rm(cwd, { recursive: true, force: true });
    }
  });
}

test("failed provisional MCP disposal releases the fork lease after a child checkpoint failure", { timeout: 10_000 }, async () => {
  const cwd = await fs.mkdtemp(join(tmpdir(), "glm-fork-disposal-lease-"));
  const dir = join(cwd, "sessions");
  const store = new SessionStore(dir);
  const disposalFailure = new Error("fixture provisional disposal failed");
  const childTools = new SessionMcpTools([]);
  const dispose = childTools.dispose.bind(childTools);
  let childDisposals = 0;
  childTools.dispose = async () => {
    childDisposals++;
    await dispose();
    throw disposalFailure;
  };
  let connections = 0;
  const agent = new GlmAcpAgent(connection() as never, {
    sessionStore: store, glm: textGlm(),
    connectSessionMcpServers: async () => ++connections === 2 ? childTools : new SessionMcpTools([]),
  });
  try {
    const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
    let checkpointFailure: unknown;
    await withCheckpointWrites(dir, async snapshot => {
      if (snapshot.sessionId !== sessionId) {
        // The actual atomic rename rejects because its destination is a directory.
        await fs.mkdir(join(dir, `${snapshot.sessionId}.json`));
      }
    }, async () => {
      const forkError = await agent.unstable_forkSession({ sessionId, cwd, mcpServers: [] }).then(
        () => { throw new Error("fixture checkpoint must reject"); },
        error => error as Error,
      );
      // Prove the release through public behavior before checking error shape.
      assert.equal((await agent.prompt({ sessionId, prompt: [{ type: "text", text: "continue after failed cleanup" }] })).stopReason, "end_turn");
      assert.ok(forkError instanceof AggregateError);
      checkpointFailure = forkError.errors[0];
      assert.match((checkpointFailure as NodeJS.ErrnoException).syscall ?? "", /rename/);
      assert.equal(forkError.errors[1], disposalFailure);
      assert.equal(forkError.cause, disposalFailure);
      assert.equal(childDisposals, 1);
    });
    await assert.rejects(store.flush(), error => error instanceof AggregateError && error.errors.includes(checkpointFailure));
  } finally {
    await store.flush().catch(() => undefined);
    await agent.shutdown("disconnect");
    await fs.rm(cwd, { recursive: true, force: true });
  }
});

test("a duplicate configuration save in the same clock tick cannot overwrite restored directory context", { timeout: 10_000 }, async t => {
  const cwd = await fs.mkdtemp(join(tmpdir(), "glm-restore-duplicate-config-"));
  const replacementCwd = join(cwd, "replacement");
  const dir = join(cwd, "sessions");
  await fs.mkdir(replacementCwd);
  await fs.writeFile(join(cwd, "AGENTS.md"), "ORIGINAL_DIRECTORY_CONTEXT");
  await fs.writeFile(join(replacementCwd, "AGENTS.md"), "RESTORED_DIRECTORY_CONTEXT");
  t.mock.timers.enable({ apis: ["Date"], now: new Date("2026-10-01T10:00:00.000Z") });
  const store = new SessionStore(dir);
  const agent = new GlmAcpAgent(connection() as never, { sessionStore: store, glm: textGlm() });
  const written = deferred();
  const release = deferred();
  let restoring: Promise<unknown> | undefined;
  let setting: Promise<unknown> | undefined;
  let held = false;
  try {
    const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
    await agent.setSessionMode({ sessionId, modeId: "default" });
    await withCheckpointWrites(dir, async snapshot => {
      if (snapshot.cwd !== replacementCwd || held) return;
      held = true;
      written.resolve();
      await release.promise;
    }, async () => {
      restoring = agent.resumeSession({ sessionId, cwd: replacementCwd, mcpServers: [] });
      await written.promise;
      // This queues a complete original snapshot behind the held replacement
      // write, despite changing neither configuration nor updatedAt.
      setting = agent.setSessionMode({ sessionId, modeId: "default" });
      release.resolve();
      await Promise.all([restoring, setting]);
      await store.flush();
      const persisted = await new SessionStore(dir).loadAsync(sessionId);
      assert.equal(persisted?.cwd, replacementCwd);
      const system = persisted?.messages.find(message => message.role === "system");
      assert.match(String(system?.content), /RESTORED_DIRECTORY_CONTEXT/);
      assert.doesNotMatch(String(system?.content), /ORIGINAL_DIRECTORY_CONTEXT/);
      const listed = await agent.listSessions({});
      assert.equal(listed.sessions.find(session => session.sessionId === sessionId)?.cwd, replacementCwd);
    });
  } finally {
    release.resolve();
    await Promise.allSettled([restoring, setting]);
    t.mock.timers.reset();
    await agent.shutdown("disconnect");
    await fs.rm(cwd, { recursive: true, force: true });
  }
});

test("restore retains both required checkpoint failures when provisional disposal also fails", { timeout: 10_000 }, async () => {
  const cwd = await fs.mkdtemp(join(tmpdir(), "glm-restore-all-errors-"));
  const dir = join(cwd, "sessions");
  const backup = join(cwd, "saved-sessions");
  const store = new SessionStore(dir);
  const provisional = new SessionMcpTools([]);
  const disposalFailure = new Error("fixture provisional restore disposal failed");
  const dispose = provisional.dispose.bind(provisional);
  let disposals = 0;
  provisional.dispose = async () => {
    disposals++;
    await dispose();
    throw disposalFailure;
  };
  let connections = 0;
  let faultInstalled = false;
  const agent = new GlmAcpAgent(connection() as never, {
    sessionStore: store, glm: textGlm(),
    connectSessionMcpServers: async () => {
      if (++connections !== 2) return new SessionMcpTools([]);
      // Both the replacement and recovery checkpoint hit a real mkdir failure.
      await fs.rename(dir, backup);
      await fs.writeFile(dir, "a file occupies the session directory");
      faultInstalled = true;
      return provisional;
    },
  });
  try {
    const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "seed history" }] });
    const restoreError = await agent.resumeSession({ sessionId, cwd, mcpServers: [] }).then(
      () => { throw new Error("fixture checkpoint must reject"); },
      error => error as Error,
    );
    await fs.rm(dir);
    await fs.rename(backup, dir);
    faultInstalled = false;
    // Public follow-up proves cleanup still releases the lease and retains
    // the original; restoring storage must make it usable without reconnecting.
    assert.equal((await agent.prompt({ sessionId, prompt: [{ type: "text", text: "continue" }] })).stopReason, "end_turn");
    assert.equal(connections, 2);
    assert.equal(disposals, 1);
    assert.ok(restoreError instanceof AggregateError);
    const checkpointErrors = restoreError.errors[0];
    assert.ok(checkpointErrors instanceof AggregateError);
    assert.equal(checkpointErrors.errors.length, 2);
    for (const failure of checkpointErrors.errors) {
      assert.equal((failure as NodeJS.ErrnoException).code, "EEXIST");
    }
    assert.notEqual(checkpointErrors.errors[0], checkpointErrors.errors[1]);
    assert.equal(restoreError.errors[1], disposalFailure);
    assert.equal(restoreError.cause, disposalFailure);
    await assert.rejects(store.flush(), error => error instanceof AggregateError &&
      error.errors.includes(checkpointErrors.errors[0]));
  } finally {
    if (faultInstalled) {
      await fs.rm(dir);
      await fs.rename(backup, dir);
    }
    await store.flush().catch(() => undefined);
    await agent.shutdown("disconnect");
    await fs.rm(cwd, { recursive: true, force: true });
  }
});

for (const mode of ["restore", "fork"]) {
  test(`a zero-timeout ${mode} assigns its required deferred checkpoint failure to the prompt`, { timeout: 10_000 }, async () => {
    const fixture = fileURLToPath(new URL("./fixtures/deferred-checkpoint.js", import.meta.url));
    const result = await promisify(execFile)(process.execPath, ["--unhandled-rejections=strict", fixture, mode], { timeout: 8_000 });
    assert.match(result.stdout, /checkpoint error reached prompt; transition released; no unhandled rejection/);
  });
}

test("successful removal retains an earlier accepted write failure for flush", async () => {
  const cwd = await fs.mkdtemp(join(tmpdir(), "glm-remove-save-failure-"));
  const dir = join(cwd, "sessions");
  const store = new SessionStore(dir);
  const session: PersistedSession = {
    sessionId: "provisional-child", cwd, messages: [], title: null,
    updatedAt: "2026-10-01T10:00:00.000Z", model: "glm-5.3", mode: "default",
  };
  try {
    const target = join(dir, `${session.sessionId}.json`);
    await fs.mkdir(target, { recursive: true });
    let writeFailure: unknown;
    await assert.rejects(store.save(session), error => { writeFailure = error; return true; });
    await fs.rm(target, { recursive: true });
    await store.save(session);
    await store.remove(session.sessionId);
    assert.equal(await store.loadAsync(session.sessionId), undefined);
    await assert.rejects(store.flush(), error => error instanceof AggregateError && error.errors.includes(writeFailure));
    await store.flush();
  } finally {
    await fs.rm(cwd, { recursive: true, force: true });
  }
});

test("loads and flush observe an accepted removal after its preceding checkpoint", async () => {
  const cwd = await fs.mkdtemp(join(tmpdir(), "glm-remove-read-order-"));
  const store = new SessionStore(join(cwd, "sessions"));
  const session: PersistedSession = {
    sessionId: "provisional-child", cwd, messages: [], title: null,
    updatedAt: "2026-10-01T10:00:00.000Z", model: "glm-5.3", mode: "default",
  };
  try {
    const saving = store.save(session);
    const removing = store.remove(session.sessionId);
    assert.equal(await store.loadAsync(session.sessionId), undefined);
    await store.flush();
    await Promise.all([saving, removing]);
    await assert.rejects(fs.access(join(cwd, "sessions", `${session.sessionId}.json`)), { code: "ENOENT" });
    await store.remove(session.sessionId); // An already absent provisional child is safe to remove.
    await store.save({ ...session, title: "explicit later write" });
    assert.equal((await store.loadAsync(session.sessionId))?.title, "explicit later write");
  } finally {
    await fs.rm(cwd, { recursive: true, force: true });
  }
});
