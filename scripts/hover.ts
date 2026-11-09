// Hover a type through the tsrs language server, exactly like an editor does.
//
//   node scripts/hover.ts examples/hover.ts Text [Greedy ...]
//
// Starts `tsrs --lsp --stdio` (via the tsdk launcher VS Code uses, see
// .vscode/settings.json), opens the file, sends textDocument/hover on the
// type's name and prints the hover text and how long it took. Several names
// are hovered one after another in the same server session, like an editor
// that stays open.
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = resolve(import.meta.dirname, "..");

export interface HoverResult {
  text: string;
  seconds: number;
}

/** Give up after this long (a crashed or stuck server would otherwise hang the caller). */
const TIMEOUT_MS = Number(process.env.HOVER_TIMEOUT_MS ?? 300_000);

export const hover = async (file: string, name: string): Promise<HoverResult> =>
  (await hoverAll(file, [name]))[0]!;

/** Hover each name in turn, in one language-server session. */
export async function hoverAll(file: string, names: string[]): Promise<HoverResult[]> {
  const path = resolve(file);
  const source = readFileSync(path, "utf8");
  const lines = source.split("\n");
  const positions = names.map((name) => {
    const line = lines.findIndex((l) => new RegExp(`\\btype ${name}\\b`).test(l));
    if (line < 0) throw new Error(`no "type ${name}" in ${file}`);
    return { line, character: lines[line]!.indexOf(name) + 1 };
  });
  const uri = pathToFileURL(path).href;

  const server = spawn(resolve(ROOT, process.env.TSDK ?? "tsdk/tsgo"), ["--lsp", "--stdio"], {
    cwd: ROOT,
  });
  const failed = new Promise<never>((_, reject) => {
    server.on("error", reject);
    server.on("exit", (code, signal) =>
      reject(new Error(`the language server exited (code ${code}, signal ${signal})`)),
    );
    setTimeout(
      () => reject(new Error(`no hover result after ${TIMEOUT_MS / 1000}s`)),
      TIMEOUT_MS,
    ).unref();
  });
  failed.catch(() => {}); // only surfaces through the races below
  let buffer = Buffer.alloc(0);
  let nextId = 0;
  const pending = new Map<
    number,
    (message: { result?: { contents?: { value?: string } } }) => void
  >();
  const send = (message: object) => {
    const body = JSON.stringify({ jsonrpc: "2.0", ...message });
    server.stdin.write(`Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`);
  };
  const request = (method: string, params: object) =>
    Promise.race([
      failed,
      new Promise<{ result?: { contents?: { value?: string } } }>((done) => {
        const id = ++nextId;
        pending.set(id, done);
        send({ id, method, params });
      }),
    ]);
  server.stdout.on("data", (chunk: Buffer) => {
    buffer = Buffer.concat([buffer, chunk]);
    for (;;) {
      const end = buffer.indexOf("\r\n\r\n");
      if (end < 0) return;
      const length = Number(/Content-Length: (\d+)/.exec(buffer.subarray(0, end).toString())![1]);
      if (buffer.length < end + 4 + length) return;
      const message = JSON.parse(buffer.subarray(end + 4, end + 4 + length).toString());
      buffer = buffer.subarray(end + 4 + length);
      if (message.id !== undefined && pending.has(message.id)) {
        pending.get(message.id)!(message);
        pending.delete(message.id);
      } else if (message.id !== undefined && message.method) {
        send({ id: message.id, result: null }); // answer server->client requests
      }
    }
  });

  try {
    await request("initialize", {
      processId: process.pid,
      rootUri: pathToFileURL(ROOT).href,
      capabilities: {},
    });
    send({ method: "initialized", params: {} });
    send({
      method: "textDocument/didOpen",
      params: { textDocument: { uri, languageId: "typescript", version: 1, text: source } },
    });
    const results: HoverResult[] = [];
    for (const position of positions) {
      const started = performance.now();
      const response = await request("textDocument/hover", { textDocument: { uri }, position });
      const seconds = (performance.now() - started) / 1000;
      results.push({ text: response.result?.contents?.value ?? "", seconds });
    }
    return results;
  } finally {
    server.removeAllListeners("exit");
    server.kill();
  }
}

if (import.meta.main) {
  const [file, ...names] = process.argv.slice(2);
  if (!file || names.length === 0) {
    console.error("usage: node scripts/hover.ts <file> <TypeName> [<TypeName> ...]");
    process.exit(2);
  }
  for (const { text, seconds } of await hoverAll(file, names)) {
    console.log(text);
    console.log(`(${seconds.toFixed(2)}s)`);
  }
}
