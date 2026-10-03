import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { closeSync, constants, mkdtempSync, openSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GlmAcpAgent } from "../protocol/agent.js";
import { PROTOCOL_VERSION } from "@agentclientprotocol/sdk";
import { readLocalTextFileBounded, readLocalTextPage } from "../tools/file-reader.js";

const isolatedHome = mkdtempSync(join(tmpdir(), "glm-bounded-read-home-"));
process.env["HOME"] = isolatedHome;
process.env["USERPROFILE"] = isolatedHome;
test.after(() => rmSync(isolatedHome, { recursive: true, force: true }));

// A parent-side deadline keeps a regressed synchronous read or blocked libuv
// worker from hanging the test runner itself. The child exercises real I/O.
async function assertChildCompletes(source: string): Promise<void> {
  const child = spawn(process.execPath, ["--input-type=module", "--eval", source], {
    stdio: ["ignore", "pipe", "pipe"],
  });
  let output = "";
  child.stdout.on("data", chunk => { output += String(chunk); });
  child.stderr.on("data", chunk => { output += String(chunk); });
  let timedOut = false;
  const deadline = setTimeout(() => { timedOut = true; child.kill("SIGKILL"); }, 2500);
  try {
    const result = await new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve, reject) => {
      child.once("error", reject);
      child.once("exit", (code, signal) => resolve({ code, signal }));
    });
    assert.equal(timedOut, false, `file read blocked past the parent deadline: ${output}`);
    assert.equal(result.code, 0, `child exited ${result.code}/${result.signal}: ${output}`);
  } finally {
    clearTimeout(deadline);
    if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
  }
}

const readerUrl = new URL("../tools/file-reader.js", import.meta.url).href;
const agentUrl = new URL("../protocol/agent.js", import.meta.url).href;

for (const idleWriter of [false, true]) {
  for (const reader of ["page", "full"] as const) {
    test(`${reader} reader promptly rejects ${idleWriter ? "connected idle" : "writerless"} FIFOs despite cancellation`, { skip: process.platform === "win32" }, async () => {
      const dir = mkdtempSync(join(tmpdir(), "glm-fifo-reader-"));
      const path = join(dir, "pipe.txt");
      let writer: number | undefined;
      try {
        execFileSync("mkfifo", [path]);
        if (idleWriter) writer = openSync(path, constants.O_RDWR | constants.O_NONBLOCK);
        await assertChildCompletes(`
          import assert from "node:assert/strict";
          import { readLocalTextPage, readLocalTextFileBounded } from ${JSON.stringify(readerUrl)};
          const controller = new AbortController();
          const abort = setTimeout(() => controller.abort(), 20);
          const path = ${JSON.stringify(path)};
          const read = ${reader === "page" ? "readLocalTextPage(path, 1, 10, 1024, controller.signal)" : "readLocalTextFileBounded(path, 1024, controller.signal)"};
          try { await assert.rejects(read, /regular file|aborted/i); }
          finally { clearTimeout(abort); }
        `);
      } finally {
        if (writer !== undefined) closeSync(writer);
        rmSync(dir, { recursive: true, force: true });
      }
    });
  }
}

test("local reader rejects a FIFO substituted after regular-file preflight", { skip: process.platform === "win32" }, async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-reader-swap-"));
  const path = join(dir, "swapped.txt");
  writeFileSync(path, "ordinary file");
  try {
    await assertChildCompletes(`
      import assert from "node:assert/strict";
      import fs from "node:fs";
      import { execFileSync } from "node:child_process";
      import { syncBuiltinESMExports } from "node:module";
      import { readLocalTextPage } from ${JSON.stringify(readerUrl)};
      const originalOpen = fs.promises.open;
      fs.promises.open = async (...args) => {
        fs.unlinkSync(${JSON.stringify(path)});
        execFileSync("mkfifo", [${JSON.stringify(path)}]);
        return originalOpen(...args);
      };
      syncBuiltinESMExports();
      await assert.rejects(readLocalTextPage(${JSON.stringify(path)}, 1, 10, 1024), /regular file/i);
    `);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("project context skips a writerless FIFO and loads the regular CLAUDE.md fallback", { skip: process.platform === "win32" }, async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-context-fifo-"));
  writeFileSync(join(dir, "CLAUDE.md"), "FALLBACK regular context");
  try {
    execFileSync("mkfifo", [join(dir, "AGENTS.md")]);
    await assertChildCompletes(`
      import assert from "node:assert/strict";
      import { GlmAcpAgent } from ${JSON.stringify(agentUrl)};
      let prompt = "";
      const glm = { async *streamChat(messages) {
        prompt = messages[0].content;
        yield { done: true, stopReason: "stop" };
      } };
      const agent = new GlmAcpAgent({ async sessionUpdate() {} }, { glm, sessionStore: null });
      const { sessionId } = await agent.newSession({ cwd: ${JSON.stringify(dir)}, mcpServers: [] });
      await agent.prompt({ sessionId, prompt: [{ type: "text", text: "hi" }] });
      assert.match(prompt, /FALLBACK regular context/);
      await agent.shutdown("disconnect");
    `);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("project context loads asynchronously and retains only its capped prefix", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-context-async-"));
  writeFileSync(join(dir, "AGENTS.md"), `${"x".repeat(1024 * 1024)}NEVER_RETAIN_THIS_TAIL`);
  let prompt = "";
  const glm = { async *streamChat(messages: ReadonlyArray<{ content?: unknown }>) {
    prompt = String(messages[0]?.content);
    yield { done: true, stopReason: "stop" as const };
  } };
  const agent = new GlmAcpAgent({ async sessionUpdate() {} } as never, { glm, sessionStore: null });
  try {
    await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
    let yielded = false;
    const heartbeat = setImmediate(() => { yielded = true; });
    const { sessionId } = await agent.newSession({ cwd: dir, mcpServers: [] });
    clearImmediate(heartbeat);
    assert.equal(yielded, true, "session context reads must let the event loop run before completing");
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "hi" }] });
    assert.ok(prompt.includes("x".repeat(8192)));
    assert.ok(!prompt.includes("x".repeat(8193)));
    assert.doesNotMatch(prompt, /NEVER_RETAIN_THIS_TAIL/);
  } finally { await agent.shutdown("disconnect"); rmSync(dir, { recursive: true, force: true }); }
});

test("project context consumes only bounded bytes from the real async file handle", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-context-consumed-bytes-"));
  writeFileSync(join(dir, "AGENTS.md"), "context ".repeat(1024 * 1024));
  try {
    await assertChildCompletes(`
      import assert from "node:assert/strict";
      import fs from "node:fs";
      import { syncBuiltinESMExports } from "node:module";
      import { GlmAcpAgent } from ${JSON.stringify(agentUrl)};
      let consumed = 0;
      const originalOpen = fs.promises.open;
      fs.promises.open = async (...args) => {
        const handle = await originalOpen(...args);
        const originalRead = handle.read.bind(handle);
        handle.read = async (...readArgs) => {
          const result = await originalRead(...readArgs);
          consumed += result.bytesRead;
          return result;
        };
        return handle;
      };
      syncBuiltinESMExports();
      let prompt = "";
      const glm = { async *streamChat(messages) {
        prompt = messages[0].content;
        yield { done: true, stopReason: "stop" };
      } };
      const agent = new GlmAcpAgent({ async sessionUpdate() {} }, { glm, sessionStore: null });
      const { sessionId } = await agent.newSession({ cwd: ${JSON.stringify(dir)}, mcpServers: [] });
      await agent.prompt({ sessionId, prompt: [{ type: "text", text: "hi" }] });
      assert.match(prompt, /context context/);
      assert.ok(consumed > 0 && consumed <= 32768, "expected bounded async bytes, consumed " + consumed);
      await agent.shutdown("disconnect");
    `);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("both local readers reject directories as non-regular files", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-reader-directory-"));
  try {
    await assert.rejects(readLocalTextPage(dir, 1, 10, 1024), /regular file/i);
    await assert.rejects(readLocalTextFileBounded(dir, 1024), /regular file/i);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("both local readers reject device inputs", { skip: process.platform === "win32" }, async () => {
  await assert.rejects(readLocalTextPage("/dev/null", 1, 10, 1024), /regular file/i);
  await assert.rejects(readLocalTextFileBounded("/dev/null", 1024), /regular file/i);
});

test("project context character truncation does not retain half a UTF-8 character", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-context-utf8-"));
  writeFileSync(join(dir, "AGENTS.md"), `${"x".repeat(8191)}🙂tail`);
  let prompt = "";
  const glm = { async *streamChat(messages: ReadonlyArray<{ content?: unknown }>) {
    prompt = String(messages[0]?.content);
    yield { done: true, stopReason: "stop" as const };
  } };
  const agent = new GlmAcpAgent({ async sessionUpdate() {} } as never, { glm, sessionStore: null });
  try {
    const { sessionId } = await agent.newSession({ cwd: dir, mcpServers: [] });
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "hi" }] });
    assert.ok(prompt.includes("x".repeat(8191)));
    assert.ok(!prompt.includes("\ud83d"), "a capped project context must omit an incomplete surrogate pair");
    assert.doesNotMatch(prompt, /tail/);
  } finally { await agent.shutdown("disconnect"); rmSync(dir, { recursive: true, force: true }); }
});

test("local readers reject cancellation before attempting a missing path", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-reader-abort-"));
  const controller = new AbortController();
  controller.abort();
  try {
    const path = join(dir, "missing.txt");
    await assert.rejects(readLocalTextPage(path, 1, 10, 1024, controller.signal), /aborted/i);
    await assert.rejects(readLocalTextFileBounded(path, 1024, controller.signal), /aborted/i);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

for (const reader of ["page", "full", "prefix"] as const) {
  test(`${reader} reader discards a final read cancelled while awaiting file I/O`, async () => {
    const dir = mkdtempSync(join(tmpdir(), "glm-reader-final-abort-"));
    const path = join(dir, "file.txt");
    writeFileSync(path, reader === "full" ? "" : "hello");
    const expression = reader === "page"
      ? "readLocalTextPage(path, 1, 10, 5, controller.signal)"
      : reader === "full"
        ? "readLocalTextFileBounded(path, 16, controller.signal)"
        : "readLocalTextPrefix(path, 5, controller.signal)";
    try {
      await assertChildCompletes(`
        import assert from "node:assert/strict";
        import fs from "node:fs";
        import { syncBuiltinESMExports } from "node:module";
        import { readLocalTextPage, readLocalTextFileBounded, readLocalTextPrefix } from ${JSON.stringify(readerUrl)};
        const controller = new AbortController();
        const originalOpen = fs.promises.open;
        fs.promises.open = async (...args) => {
          const handle = await originalOpen(...args);
          const originalRead = handle.read.bind(handle);
          handle.read = async (...readArgs) => {
            const result = await originalRead(...readArgs);
            controller.abort();
            return result;
          };
          return handle;
        };
        syncBuiltinESMExports();
        const path = ${JSON.stringify(path)};
        await assert.rejects(${expression}, /aborted/i);
      `);
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });
}

test("regular-file symlinks preserve local page, UTF-8, and full-file budgets", async t => {
  const dir = mkdtempSync(join(tmpdir(), "glm-reader-symlink-"));
  const target = join(dir, "target.txt");
  const path = join(dir, "link.txt");
  writeFileSync(target, "one\n🙂\nlast");
  try {
    try { symlinkSync(target, path, "file"); }
    catch (error) {
      if (process.platform === "win32" && (error as NodeJS.ErrnoException).code === "EPERM") {
        t.skip("Windows file symlinks need Developer Mode or elevation");
        return;
      }
      throw error;
    }
    assert.equal((await readLocalTextPage(path, 2, 1, 100)).text, "🙂");
    assert.equal(await readLocalTextFileBounded(path, 13), "one\n🙂\nlast");
    await assert.rejects(readLocalTextFileBounded(path, 12), /exceeds.*12-byte/);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
