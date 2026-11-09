/**
 * Plain float64 GPT-2 forward pass over the trained weights, used to measure
 * how closely the fixed-point model (and so the type-level one) tracks the
 * real network.
 */
import type { FloatWeights } from "./quantize.ts";

type V = number[];
const dot = (a: V, b: V) => a.reduce((s, v, i) => s + v * b[i]!, 0);
const matvec = (w: V[], x: V, b?: V) => w.map((r, j) => dot(x, r) + (b ? b[j]! : 0));
const layerNorm = (x: V, g: V, b: V) => {
  const mean = x.reduce((a, v) => a + v, 0) / x.length;
  const variance = x.reduce((a, v) => a + (v - mean) ** 2, 0) / x.length;
  return x.map((v, i) => ((v - mean) / Math.sqrt(variance + 1e-5)) * g[i]! + b[i]!);
};
const gelu = (x: number) =>
  0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
const softmax = (s: V) => {
  const m = Math.max(...s);
  const e = s.map((v) => Math.exp(v - m));
  const sum = e.reduce((a, v) => a + v, 0);
  return e.map((v) => v / sum);
};

export function generateFloat(f: FloatWeights, prompt: number[], steps: number): number[] {
  const { n_head, n_embd, block_size, n_layer } = f.config;
  const w = f.weights as Record<string, V & V[]>;
  const hd = n_embd / n_head;
  const tokens = [...prompt];
  const cache = Array.from({ length: n_layer }, () => ({ k: [] as V[], v: [] as V[] }));

  for (let pos = 0; pos < tokens.length && pos < block_size; pos++) {
    let x = (w["wte.weight"]![tokens[pos]!] as V).map(
      (v, i) => v + (w["wpe.weight"]![pos] as V)[i]!,
    );
    for (let l = 0; l < n_layer; l++) {
      const p = (k: string) => w[`h.${l}.${k}`]!;
      const qkv = matvec(
        p("c_attn.weight"),
        layerNorm(x, p("ln_1.weight"), p("ln_1.bias")),
        p("c_attn.bias"),
      );
      const q = qkv.slice(0, n_embd);
      cache[l]!.k.push(qkv.slice(n_embd, 2 * n_embd));
      cache[l]!.v.push(qkv.slice(2 * n_embd));
      const out: V = [];
      for (let h = 0; h < n_head; h++) {
        const sl = (v: V) => v.slice(h * hd, (h + 1) * hd);
        const a = softmax(cache[l]!.k.map((k) => dot(sl(q), sl(k)) / Math.sqrt(hd)));
        for (let d = 0; d < hd; d++)
          out.push(
            dot(
              a,
              cache[l]!.v.map((v) => v[h * hd + d]!),
            ),
          );
      }
      const attn = matvec(p("c_proj.weight"), out, p("c_proj.bias"));
      x = x.map((v, i) => v + attn[i]!);
      const hid = matvec(
        p("c_fc.weight"),
        layerNorm(x, p("ln_2.weight"), p("ln_2.bias")),
        p("c_fc.bias"),
      ).map(gelu);
      const y = matvec(p("mlp_proj.weight"), hid, p("mlp_proj.bias"));
      x = x.map((v, i) => v + y[i]!);
    }
    if (pos === tokens.length - 1 && tokens.length - prompt.length < steps) {
      const logits = matvec(w["wte.weight"]!, layerNorm(x, w["ln_f.weight"]!, w["ln_f.bias"]!));
      tokens.push(logits.indexOf(Math.max(...logits)));
    }
  }
  return tokens;
}
