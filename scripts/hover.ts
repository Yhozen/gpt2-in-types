// Hover a type through the tsrs language server, exactly like an editor does.
//
//   node scripts/hover.ts examples/hover.ts Text
//
// Starts `tsrs --lsp --stdio` (via the tsdk launcher VS Code uses, see
// .vscode/settings.json), opens the file, sends textDocument/hover on the
// type's name and prints the hover text and how long it took.
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const ROOT = resolve(import.meta.dirname, "..");

export interface HoverResult {
  text: string;
  seconds: number;
}

export async function hover(file: string, name: string): Promise<HoverResult> {
  const path = resolve(file);
  const source = readFileSync(path, "utf8");
  const lines = source.split("\n");
  const line = lines.findIndex((l) => new RegExp(`\\btype ${name}\\b`).test(l));
  if (line < 0) throw new Error(`no "type ${name}" in ${file}`);
  const uri = pathToFileURL(path).href;

  const server = spawn(resolve(ROOT, process.env.TSDK ?? "tsdk/tsgo"), ["--lsp", "--stdio"], {
    cwd: ROOT,
  });
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
    new Promise<{ result?: { contents?: { value?: string } } }>((done) => {
      const id = ++nextId;
      pending.set(id, done);
      send({ id, method, params });
    });
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
    const started = performance.now();
    const response = await request("textDocument/hover", {
      textDocument: { uri },
      position: { line, character: lines[line]!.indexOf(name) + 1 },
    });
    const seconds = (performance.now() - started) / 1000;
    return { text: response.result?.contents?.value ?? "", seconds };
  } finally {
    server.kill();
  }
}

if (import.meta.main) {
  const [file, name] = process.argv.slice(2);
  if (!file || !name) {
    console.error("usage: node scripts/hover.ts <file> <TypeName>");
    process.exit(2);
  }
  const { text, seconds } = await hover(file, name);
  console.log(text);
  console.log(`(${seconds.toFixed(2)}s)`);
}
