import { constants } from "node:fs";
import { open, stat, type FileHandle } from "node:fs/promises";
import { StringDecoder } from "node:string_decoder";

/** Follow ordinary file symlinks, but never read a FIFO, directory, or device. */
async function openRegularFile(path: string, signal?: AbortSignal): Promise<FileHandle> {
  if (signal?.aborted) throw new Error("The operation was aborted");
  // Avoid opening known devices and special Windows paths. This preflight is
  // not the authority: the path may change before open(), so validate the
  // opened handle too. POSIX nonblocking open keeps a swapped FIFO from
  // trapping a filesystem worker before that validation can run.
  if (!(await stat(path)).isFile()) throw new Error("text reads require a regular file");
  if (signal?.aborted) throw new Error("The operation was aborted");
  const flags = process.platform === "win32"
    ? constants.O_RDONLY
    : constants.O_RDONLY | constants.O_NONBLOCK | constants.O_NOCTTY;
  const handle = await open(path, flags);
  try {
    if (!(await handle.stat()).isFile()) throw new Error("text reads require a regular file");
    if (signal?.aborted) throw new Error("The operation was aborted");
    return handle;
  } catch (error) {
    await handle.close();
    throw error;
  }
}

export interface TextPage {
  text: string;
  firstLine: number;
  lastCompleteLine: number;
  totalLines?: number;
  nextLine?: number;
  truncated: boolean;
  incompleteLine?: number;
  eof?: boolean;
}

/**
 * Read at most maxReadBytes from disk. The byte bound applies to bytes consumed
 * from the handle too, rather than trusting a pre-read stat result.
 */
export async function readLocalTextPage(
  path: string,
  offset: number,
  limit: number,
  maxReadBytes: number,
  signal?: AbortSignal,
): Promise<TextPage> {
  const handle = await openRegularFile(path, signal);
  try {
    const chunks: Buffer[] = [];
    let consumed = 0;
    let eof = false;
    while (consumed < maxReadBytes) {
      if (signal?.aborted) throw new Error("The operation was aborted");
      const buffer = Buffer.allocUnsafe(Math.min(64 * 1024, maxReadBytes - consumed));
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, null);
      if (signal?.aborted) throw new Error("The operation was aborted");
      if (bytesRead === 0) { eof = true; break; }
      chunks.push(bytesRead === buffer.length ? buffer : Buffer.from(buffer.subarray(0, bytesRead)));
      consumed += bytesRead;
    }
    const bytes = Buffer.concat(chunks, consumed);
    const safeOffset = Math.max(1, Math.floor(offset));
    const safeLimit = Math.max(1, Math.floor(limit));
    const selectedLines: Buffer[] = [];
    let completeLineCount = 0;
    let lineStart = 0;
    for (let i = 0; i < bytes.length; i++) {
      if (bytes[i] === 0x0a) {
        completeLineCount += 1;
        if (completeLineCount >= safeOffset && completeLineCount < safeOffset + safeLimit) {
          selectedLines.push(bytes.subarray(lineStart, i));
        }
        lineStart = i + 1;
      }
    }
    const hasPartial = lineStart < bytes.length;
    if (eof && hasPartial) {
      completeLineCount += 1;
      if (completeLineCount >= safeOffset && completeLineCount < safeOffset + safeLimit) {
        selectedLines.push(bytes.subarray(lineStart));
      }
    }
    const totalLines = eof ? completeLineCount : undefined;
    if (totalLines !== undefined && safeOffset > totalLines) {
      return { text: "", firstLine: safeOffset, lastCompleteLine: totalLines, totalLines, truncated: false };
    }
    const selected = selectedLines.map(line => decodeUtf8Safely(line));
    const lastCompleteLine = !eof && safeOffset > completeLineCount
      ? completeLineCount
      : safeOffset + selected.length - 1;
    const currentLine = completeLineCount + 1;
    const pageEndsAtKnownPartial = !eof && hasPartial && safeOffset <= currentLine && safeOffset + safeLimit - 1 >= currentLine;
    const pageEndsAtBudget = !eof && (hasPartial || bytes.length === maxReadBytes);
    const pageEnd = safeOffset + selected.length - 1;
    const nextLine = !pageEndsAtKnownPartial && selected.length === safeLimit && pageEnd < completeLineCount
      ? pageEnd + 1
      : undefined;
    return {
      text: selected.join("\n") + (pageEndsAtKnownPartial && selected.length === 0 ? decodeUtf8Safely(bytes.subarray(lineStart)) : ""),
      firstLine: safeOffset,
      lastCompleteLine,
      ...(totalLines === undefined ? {} : { totalLines }),
      ...(nextLine === undefined ? {} : { nextLine }),
      truncated: pageEndsAtBudget,
      ...(pageEndsAtKnownPartial ? { incompleteLine: currentLine } : {}),
    };
  } finally {
    await handle.close();
  }
}

/** Read a complete local file only when it stays inside the configured edit budget. */
export async function readLocalTextFileBounded(path: string, maxReadBytes: number, signal?: AbortSignal): Promise<string> {
  const handle = await openRegularFile(path, signal);
  try {
    const chunks: Buffer[] = [];
    let consumed = 0;
    while (true) {
      if (signal?.aborted) throw new Error("The operation was aborted");
      const buffer = Buffer.allocUnsafe(Math.min(64 * 1024, maxReadBytes + 1 - consumed));
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, null);
      if (signal?.aborted) throw new Error("The operation was aborted");
      if (bytesRead === 0) return decodeUtf8Safely(Buffer.concat(chunks, consumed));
      consumed += bytesRead;
      if (consumed > maxReadBytes) {
        throw new Error(`file exceeds the ${maxReadBytes}-byte read/edit limit`);
      }
      chunks.push(bytesRead === buffer.length ? buffer : Buffer.from(buffer.subarray(0, bytesRead)));
    }
  } finally {
    await handle.close();
  }
}

/** Keep only a bounded UTF-8 prefix for optional project context. */
export async function readLocalTextPrefix(path: string, maxReadBytes: number, signal?: AbortSignal): Promise<string> {
  const handle = await openRegularFile(path, signal);
  try {
    const chunks: Buffer[] = [];
    let consumed = 0;
    while (consumed < maxReadBytes) {
      if (signal?.aborted) throw new Error("The operation was aborted");
      const buffer = Buffer.allocUnsafe(Math.min(64 * 1024, maxReadBytes - consumed));
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, null);
      if (signal?.aborted) throw new Error("The operation was aborted");
      if (bytesRead === 0) break;
      chunks.push(bytesRead === buffer.length ? buffer : Buffer.from(buffer.subarray(0, bytesRead)));
      consumed += bytesRead;
    }
    return decodeUtf8Safely(Buffer.concat(chunks, consumed));
  } finally {
    await handle.close();
  }
}

function decodeUtf8Safely(bytes: Buffer): string {
  // StringDecoder retains an incomplete trailing sequence instead of emitting
  // U+FFFD; by not calling end(), the bounded scan drops that incomplete tail.
  return new StringDecoder("utf8").write(bytes);
}
