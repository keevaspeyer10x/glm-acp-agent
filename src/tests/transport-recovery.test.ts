import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import OpenAI from "openai";
import { GlmAcpAgent } from "../protocol/agent.js";
import { SessionStore } from "../protocol/session-store.js";
import type { GlmStreamChunk } from "../llm/glm-client.js";

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
