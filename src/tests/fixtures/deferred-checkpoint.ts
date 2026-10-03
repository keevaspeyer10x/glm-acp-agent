import assert from "node:assert/strict";
import { mkdtemp, rename, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GlmAcpAgent } from "../../protocol/agent.js";
import { SessionStore } from "../../protocol/session-store.js";
import type { GlmStreamChunk } from "../../llm/glm-client.js";

// This runs under Node's default rejection policy: a detached checkpoint error
// must fail the process, even if both public requests have already settled.
const cwd = await mkdtemp(join(tmpdir(), "glm-checkpoint-owner-"));
const dir = join(cwd, "sessions");
const backup = join(cwd, "saved-sessions");
const store = new SessionStore(dir);
let ready!: () => void;
const providerWaiting = new Promise<void>(resolve => { ready = resolve; });
let calls = 0;
const agent = new GlmAcpAgent({
  signal: new AbortController().signal,
  async sessionUpdate() {},
  async requestPermission() { return { outcome: { outcome: "selected", optionId: "allow" } }; },
} as never, {
  sessionStore: store,
  sessionDrainTimeoutMs: 0,
  glm: {
    streamChat(_messages, signal): AsyncIterable<GlmStreamChunk> {
      let nextCall = 0;
      const seed = ++calls !== 2;
      return {
        [Symbol.asyncIterator]() { return this; },
        next(): Promise<IteratorResult<GlmStreamChunk>> {
          if (++nextCall === 1) {
            return Promise.resolve({ value: { text: seed ? "seed" : "partial", done: seed, stopReason: "stop" }, done: false });
          }
          if (seed) return Promise.resolve({ value: undefined, done: true });
          return new Promise((_resolve, reject) => {
            signal?.addEventListener("abort", () => reject(new Error("provider aborted")), { once: true });
            ready();
          });
        },
      } as AsyncIterableIterator<GlmStreamChunk>;
    },
  },
});
let faultInstalled = false;
try {
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  await agent.prompt({ sessionId, prompt: [{ type: "text", text: "seed checkpoint" }] });
  const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "drain me" }] }).then(
    response => ({ response, error: undefined }),
    error => ({ response: undefined, error: error as NodeJS.ErrnoException }),
  );
  await providerWaiting;
  await rename(dir, backup);
  await writeFile(dir, "a file occupies the session directory");
  faultInstalled = true;
  const transition = (process.argv[2] === "fork"
    ? agent.unstable_forkSession({ sessionId, cwd, mcpServers: [] })
    : agent.resumeSession({ sessionId, cwd, mcpServers: [] })).then(
    () => undefined,
    error => error as Error,
  );
  const [promptResult, transitionError] = await Promise.all([prompt, transition]);
  // Yield a turn so Node can report any detached rejection before assertions.
  await new Promise<void>(resolve => setImmediate(resolve));
  assert.equal(promptResult.error?.code, "EEXIST", "the draining prompt owns the required checkpoint failure");
  assert.match(transitionError?.message ?? "", /timed out waiting for prompt cleanup/);
  await assert.rejects(store.flush(), error => error instanceof AggregateError &&
    error.errors.some((cause: NodeJS.ErrnoException) => cause.code === "EEXIST"));
  await rm(dir);
  await rename(backup, dir);
  faultInstalled = false;
  assert.equal((await agent.prompt({ sessionId, prompt: [{ type: "text", text: "continue" }] })).stopReason, "end_turn");
  process.stdout.write("checkpoint error reached prompt; transition released; no unhandled rejection\n");
} finally {
  if (faultInstalled) {
    await rm(dir);
    await rename(backup, dir);
    await store.flush().catch(() => undefined);
  }
  await agent.shutdown("disconnect");
  await rm(cwd, { recursive: true, force: true });
}
