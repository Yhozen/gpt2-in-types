/**
 * Load the float weights written by scripts/train.py and convert them to the
 * fixed-point model (integers at scale 10^3) that both the reference and the
 * type-level implementation run.
 */
import { readFileSync } from "node:fs";

export interface Config {
  n_layer: number;
  n_head: number;
  n_embd: number;
  block_size: number;
  vocab_size: number;
}

export interface FloatWeights {
  config: Config;
  vocab: string[];
  weights: Record<string, number[] | number[][]>;
}

type Vec = bigint[];
type Mat = bigint[][];

export interface FixedLayer {
  ln1g: Vec;
  ln1b: Vec;
  attnW: Mat;
  attnB: Vec;
  projW: Mat;
  projB: Vec;
  ln2g: Vec;
  ln2b: Vec;
  fcW: Mat;
  fcB: Vec;
  mlpW: Mat;
  mlpB: Vec;
}

export interface FixedModel {
  config: Config;
  vocab: string[];
  wte: Mat;
  wpe: Mat;
  layers: FixedLayer[];
  lnfG: Vec;
  lnfB: Vec;
}

export const WEIGHTS_PATH = new URL("../../model/weights.json", import.meta.url);

export function loadFloat(path: URL | string = WEIGHTS_PATH): FloatWeights {
  return JSON.parse(readFileSync(path, "utf8")) as FloatWeights;
}

const q = (v: number): bigint => BigInt(Math.round(v * 1000));

export function quantize(f: FloatWeights): FixedModel {
  const { config, weights: w } = f;
  const hd = config.n_embd / config.n_head;
  const qScale = 1 / Math.sqrt(hd);
  const vec = (k: string) => (w[k] as number[]).map(q);
  const mat = (k: string) => (w[k] as number[][]).map((r) => r.map(q));
  // Fold the 1/sqrt(head_dim) attention scale into the query rows of c_attn.
  const scaled = (k: string) =>
    (w[k] as number[][]).map((r, j) => r.map((v) => q(j < config.n_embd ? v * qScale : v)));
  const scaledB = (k: string) =>
    (w[k] as number[]).map((v, j) => q(j < config.n_embd ? v * qScale : v));

  return {
    config,
    vocab: f.vocab,
    wte: mat("wte.weight"),
    wpe: mat("wpe.weight"),
    layers: Array.from({ length: config.n_layer }, (_, l) => {
      const p = `h.${l}.`;
      return {
        ln1g: vec(`${p}ln_1.weight`),
        ln1b: vec(`${p}ln_1.bias`),
        attnW: scaled(`${p}c_attn.weight`),
        attnB: scaledB(`${p}c_attn.bias`),
        projW: mat(`${p}c_proj.weight`),
        projB: vec(`${p}c_proj.bias`),
        ln2g: vec(`${p}ln_2.weight`),
        ln2b: vec(`${p}ln_2.bias`),
        fcW: mat(`${p}c_fc.weight`),
        fcB: vec(`${p}c_fc.bias`),
        mlpW: mat(`${p}mlp_proj.weight`),
        mlpB: vec(`${p}mlp_proj.bias`),
      };
    }),
    lnfG: vec("ln_f.weight"),
    lnfB: vec("ln_f.bias"),
  };
}

export const encode = (vocab: string[], s: string): number[] =>
  s.split("").map((c) => {
    const i = vocab.indexOf(c);
    if (i < 0) throw new Error(`character ${JSON.stringify(c)} is not in the vocabulary`);
    return i;
  });

export const decode = (vocab: string[], ids: number[]): string => ids.map((i) => vocab[i]).join("");
