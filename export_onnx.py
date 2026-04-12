"""
export_onnx.py — Export fine-tuned NanoRouter to ONNX + precompute route embeddings

Outputs:
  nanorouter.onnx        (~22MB, int8 quantized, loads in browser via onnxruntime-web)
  nanorouter_routes.json (pre-computed function embeddings + schema for cosine sim in browser)

Usage:
  python export_onnx.py --checkpoint ecommerce.pt
"""

import argparse
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

DEVICE = "cpu"


class MiniLMRouter(nn.Module):
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, num_bio_tags: int = 1):
        super().__init__()
        self.backbone    = AutoModel.from_pretrained(self.MODEL_NAME)
        self.temperature = nn.Parameter(torch.tensor(14.0))
        self.ner_head    = nn.Linear(384, num_bio_tags)

    @staticmethod
    def get_tokenizer():
        return AutoTokenizer.from_pretrained(MiniLMRouter.MODEL_NAME)

    def encode(self, input_ids, attention_mask):
        out  = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        vec  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(vec, dim=-1)

    def forward(self, input_ids, attention_mask):
        """ONNX export: returns masked token embeddings + BIO slot logits."""
        out  = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        toks = out.last_hidden_state * mask
        bio  = self.ner_head(toks)
        return toks, bio


def load_checkpoint(ckpt_path: str):
    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt      = torch.load(ckpt_path, map_location=DEVICE)
    bio_tags  = ckpt.get('bio_tags', ['O'])
    tok       = MiniLMRouter.get_tokenizer()
    model     = MiniLMRouter(num_bio_tags=len(bio_tags)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    functions = ckpt["functions"]
    print(f"  Loaded — {len(functions)} functions, {len(bio_tags)} BIO tags")
    return model, tok, functions, bio_tags


def export_onnx(model, tok, out_path: str):
    print(f"\n  Exporting ONNX -> {out_path}")
    dummy_text = ["export dummy query"]
    enc = tok(dummy_text, padding=True, truncation=True,
               max_length=64, return_tensors="pt")

    torch.onnx.export(
        model,
        (enc["input_ids"], enc["attention_mask"]),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["token_embeddings", "bio_logits"],
        dynamic_axes={
            "input_ids":       {0: "batch", 1: "seq"},
            "attention_mask":  {0: "batch", 1: "seq"},
            "token_embeddings": {0: "batch", 1: "seq"},
            "bio_logits":       {0: "batch", 1: "seq"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"  Exported (fp32): {size_mb:.1f} MB")
    return out_path


def quantize_onnx(fp32_path: str, int8_path: str):
    print(f"\n  Quantizing -> {int8_path}")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
        size_mb = Path(int8_path).stat().st_size / 1e6
        print(f"  Quantized (int8): {size_mb:.1f} MB")
        return int8_path
    except ImportError:
        print("  onnxruntime not found — skipping quantization, using fp32")
        import shutil
        shutil.copy(fp32_path, int8_path)
        return int8_path


@torch.no_grad()
def precompute_routes(model, tok, functions: list, bio_tags: list, out_path: str):
    """
    Pre-compute embeddings for all function descriptions and enum param values.

    For function routing: token-level embeddings used with MaxSim.
    For param extraction: sentence-level MEAN-POOLED embeddings of synonym descriptions.
      This uses MiniLM as designed — at sentence level — so 'open', 'launch', 'boot'
      all score high against the 'on' anchor without any hardcoded aliases.
    """
    print(f"\n  Pre-computing function & param embeddings...")
    model.eval()

    def mean_pool(text: str) -> list:
        """Normalized mean-pooled embedding — the correct MiniLM representation."""
        enc = tok([text], padding=True, truncation=True,
                  max_length=32, return_tensors="pt")
        out  = model.backbone(input_ids=enc["input_ids"],
                              attention_mask=enc["attention_mask"])
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vec  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(vec, dim=-1)[0].tolist()

    routes = []
    for i, fn in enumerate(functions):
        fn_desc = fn.get("description", fn["name"])
        enc = tok([fn_desc], padding=True, truncation=True,
                  max_length=64, return_tensors="pt")
        toks_t, _ = model(enc["input_ids"], enc["attention_mask"])
        token_embs = toks_t[0].tolist()

        params_out = {}
        for pname, pdef in fn.get("parameters", {}).items():
            vals = pdef.get("values") or pdef.get("enum") or []
            val_descs = pdef.get("value_descriptions", {})

            if vals:
                sent_embs  = {}
                tok_embs   = {}

                for v in vals:
                    desc_text = val_descs.get(v) or f"{pname}: {str(v).replace('_', ' ')}"
                    sent_embs[v] = mean_pool(desc_text)

                    v_enc = tok([desc_text], padding=True, truncation=True,
                                max_length=32, return_tensors="pt")
                    v_t, _ = model(v_enc["input_ids"], v_enc["attention_mask"])
                    tok_embs[v] = v_t[0].tolist()

                params_out[pname] = {
                    **pdef,
                    "sentence_embedding": sent_embs,
                    "token_embeddings":   tok_embs,
                }
            else:
                params_out[pname] = pdef

        routes.append({
            "name":             fn["name"],
            "description":      fn_desc,
            "token_embeddings": token_embs,
            "parameters":       params_out,
        })
        print(f"    [{i+1:02d}/{len(functions)}] {fn['name']}")

    with open(out_path, "w") as f:
        json.dump({
            "model":    "NanoRouter/all-MiniLM-L6-v2",
            "dim":      384,
            "bio_tags": bio_tags,
            "routes":   routes,
        }, f, indent=2)

    print(f"  Saved: {out_path}  ({Path(out_path).stat().st_size // 1024} KB, {len(routes)} routes)")


def verify_onnx(onnx_path: str, tok, bio_tags: list):
    """Quick sanity check — run a test query through the ONNX model."""
    print(f"\n  Verifying ONNX ({onnx_path})...")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        query = "turn off the bedroom lights"
        enc   = tok([query], padding=True, truncation=True,
                    max_length=64, return_tensors="np")

        t0  = time.perf_counter()
        out = sess.run(["token_embeddings", "bio_logits"], {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })
        ms = (time.perf_counter() - t0) * 1000

        toks, bio = out[0], out[1]
        print(f"  OK | toks={toks.shape} | bio={bio.shape} ({len(bio_tags)} tags) | {ms:.1f}ms")
    except ImportError:
        print("  onnxruntime not installed — skipping verification")


def main():
    p = argparse.ArgumentParser(description="Export NanoRouter checkpoint to ONNX for browser inference")
    p.add_argument("--checkpoint", default="ecommerce.pt", help="Path to .pt checkpoint")
    p.add_argument("--schema",     default=None,           help="Schema JSON to merge current lexical/value_descriptions from")
    p.add_argument("--out-onnx",   default="nanorouter.onnx", help="Output ONNX path")
    p.add_argument("--out-routes", default="nanorouter_routes.json", help="Output routes JSON")
    p.add_argument("--no-quantize", action="store_true", help="Skip int8 quantization")
    args = p.parse_args()

    print("\n" + "="*60)
    print("  NanoRouter ONNX Export")
    print("="*60)

    model, tok, functions, bio_tags = load_checkpoint(args.checkpoint)

    schema_path = args.schema
    if schema_path is None:
        stem = Path(args.checkpoint).stem
        candidate = Path("examples") / f"{stem}.json"
        if candidate.exists():
            schema_path = str(candidate)

    if schema_path and Path(schema_path).exists():
        print(f"  Merging schema metadata from: {schema_path}")
        schema_data  = json.load(open(schema_path))
        schema_fn_map = {fn["name"]: fn for fn in schema_data.get("functions", [])}
        for fn in functions:
            sch_fn = schema_fn_map.get(fn["name"], {})
            for pname, pdef in fn.get("parameters", {}).items():
                sch_pdef = sch_fn.get("parameters", {}).get(pname, {})
                for key in ("lexical", "value_descriptions"):
                    if key in sch_pdef:
                        pdef[key] = sch_pdef[key]
    else:
        print("  No schema file found — using checkpoint parameter metadata as-is")

    fp32_path = args.out_onnx.replace(".onnx", "_fp32.onnx")
    export_onnx(model, tok, fp32_path)

    if args.no_quantize:
        final_onnx = fp32_path
    else:
        final_onnx = args.out_onnx
        quantize_onnx(fp32_path, final_onnx)
        Path(fp32_path).unlink(missing_ok=True)

    precompute_routes(model, tok, functions, bio_tags, args.out_routes)
    verify_onnx(final_onnx, tok, bio_tags)

    print("\n" + "="*60)
    print("  Done. Files to commit to GitHub:")
    print(f"    {final_onnx}           ← ONNX model (fetch in browser)")
    print(f"    {args.out_routes}   ← pre-computed embeddings + schema")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
