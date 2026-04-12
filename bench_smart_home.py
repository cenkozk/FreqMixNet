"""
bench_smart_home.py — Smart Home domain benchmark.

Compares NanoRouter (MiniLM-L6 fine-tuned, 22 MB ONNX) vs
FunctionGemma-270M (LoRA, same training data) on smart_home.json.

Methodology:
  - Both models train on IDENTICAL data (programmatic augmentation, N/fn)
  - Test set: Ollama-generated natural queries (different distribution)
  - Metrics: Fn% (function correct), Exact% (all params correct), ms/q, size
"""
import os, sys, json, re, time, random, warnings
import urllib.request
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

import torch
from torch.optim import AdamW
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import router

warnings.filterwarnings('ignore')
random.seed(42)

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
FG_BASE   = 'google/functiongemma-270m-it'
FG_LORA   = 'fg_lora_smart_home'
SCHEMA    = 'examples/smart_home.json'
TEST_FILE = 'examples/test_smart_home.json'
TRAIN_N   = 15
FG_EPOCHS = 12
FG_LR     = 2e-4
FG_BATCH  = 4
NR_EPOCHS = 50
NR_LR     = 2e-4
OL_MODEL  = 'qwen3.5:4b'

def _ollama_online() -> bool:
    try:
        urllib.request.urlopen('http://localhost:11434/api/tags', timeout=3)
        return True
    except Exception:
        return False

def ollama(prompt: str, timeout: int = 60) -> str:
    payload = json.dumps({
        'model': OL_MODEL, 'prompt': prompt, 'stream': False,
        'options': {'temperature': 0.8, 'num_predict': 256, 'num_ctx': 512}
    }).encode()
    req = urllib.request.Request(
        'http://localhost:11434/api/generate',
        data=payload, headers={'Content-Type': 'application/json'}, method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = json.load(r).get('response', '')
            return re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    except Exception:
        return ''

def parse_list(text: str, n: int) -> list[str]:
    for m in re.finditer(r'\[.*?\]', text, re.DOTALL):
        try:
            items = json.loads(m.group())
            clean = [str(x).strip().strip('"') for x in items if len(str(x).strip()) > 5]
            if clean:
                return clean[:n]
        except Exception:
            pass
    items = re.findall(r'^\d+[.)]\s*(.{5,120})$', text, re.MULTILINE)
    return [x.strip('"') for x in items[:n]] if items else []

with open(SCHEMA) as f:
    schema = json.load(f)

def enum_values(pdef: dict) -> list:
    return pdef.get('values') or pdef.get('enum') or []

def smart_home_combos() -> list[dict]:
    """Full cartesian product of enum params per function."""
    from itertools import product
    combos = []
    for fn in schema['functions']:
        params = fn.get('parameters', {})
        enum_params = {k: enum_values(v) for k, v in params.items() if enum_values(v)}
        if not enum_params:
            combos.append({'function': fn['name'], 'params': {}})
            continue
        keys = list(enum_params)
        for vals in product(*[enum_params[k] for k in keys]):
            combos.append({'function': fn['name'], 'params': dict(zip(keys, vals))})
    return combos

def to_tools(schema: dict) -> list:
    tools = []
    for fn in schema['functions']:
        props, req = {}, []
        for pname, pdef in fn.get('parameters', {}).items():
            enums = enum_values(pdef)
            props[pname] = {'type': 'string',
                            'description': f"One of: {', '.join(str(e) for e in enums)}" if enums else pname}
            if enums:
                props[pname]['enum'] = enums
            req.append(pname)
        tools.append({'type': 'function', 'function': {
            'name': fn['name'], 'description': fn.get('description', ''),
            'parameters': {'type': 'object', 'properties': props, 'required': req}
        }})
    return tools

def label_to_fg(label: dict) -> str:
    fn_name = label['function']
    params  = {k: v for k, v in label.items() if k != 'function'}
    inner   = ', '.join(f'{k}:<escape>{v}<escape>' for k, v in params.items())
    return f'call:{fn_name}{{{inner}}}'

def parse_fg_output(text: str) -> dict | None:
    m = re.search(r'call:(\w+)\{(.*?)\}', text, re.DOTALL)
    if m:
        result = {'function': m.group(1)}
        for pm in re.finditer(r'(\w+)[=:]["\']?([^,"\'}\s<]+)', m.group(2)):
            result[pm.group(1)] = pm.group(2).strip()
        return result
    for jm in re.finditer(r'\{[^{}]+\}', text):
        try:
            d = json.loads(jm.group())
            fn = d.pop('function', d.pop('name', None))
            if fn:
                return {'function': fn, **d}
        except Exception:
            pass
    for fn in schema['functions']:
        if fn['name'] in text:
            return {'function': fn['name']}
    return None

print('=' * 64)
print('  SMART HOME: NanoRouter vs FunctionGemma-270M (LoRA)')
print('=' * 64)
print(f'  Schema : {schema["name"]}  ({len(schema["functions"])} functions)')
print(f'  Device : {DEVICE}')
print()

print(f'\n[1/4] Building training set ({TRAIN_N}/fn programmatic)...')
train_data = router.generate_data(schema, examples_per_fn=TRAIN_N)
by_fn: dict = {}
for d in train_data:
    by_fn.setdefault(d['function'], []).append(d)
train_data = []
for docs in by_fn.values():
    train_data.extend(docs[:TRAIN_N])
random.shuffle(train_data)
print(f'  Train: {len(train_data)} queries ({TRAIN_N}/fn)')

print(f'\n[1b] Loading hand-crafted test set...')
with open(TEST_FILE) as f:
    test_data = json.load(f)
print(f'  Test:  {len(test_data)} queries (natural language, no templates)')

print(f'\n[2/4] Training NanoRouter (MiniLM-L6, ~22M params, {NR_EPOCHS} epochs)...')
t0 = time.time()
nr_model, nr_tok, nr_functions, _ = router._train_model(
    schema, epochs=NR_EPOCHS, lr=NR_LR, batch_size=16,
    extra_examples=train_data, examples_only=True
)
fn_vecs    = router._build_fn_cache(nr_model, nr_tok, nr_functions)
nr_train_s = time.time() - t0
nr_params  = sum(p.numel() for p in nr_model.parameters())
print(f'  Done in {nr_train_s:.1f}s  |  {nr_params/1e6:.1f}M params')

print(f'\n[3/4] Fine-tuning FunctionGemma-270M (LoRA r=8, {FG_EPOCHS} epochs)...')
fg_proc  = AutoProcessor.from_pretrained(FG_BASE)
base_mdl = AutoModelForCausalLM.from_pretrained(FG_BASE, torch_dtype=torch.bfloat16).to(DEVICE)
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
    target_modules=['q_proj', 'v_proj'], lora_dropout=0.05, bias='none',
)
fg_model   = get_peft_model(base_mdl, lora_cfg)
fg_params  = sum(p.numel() for p in base_mdl.parameters())
fg_trainable = sum(p.numel() for p in fg_model.parameters() if p.requires_grad)
print(f'  Base: {fg_params/1e6:.0f}M params | LoRA trainable: {fg_trainable:,}')

tools = to_tools(schema)
fg_train = []
for ex in train_data:
    label   = {'function': ex['function'], **ex.get('params', {})}
    target  = label_to_fg(label)
    msgs    = [{'role': 'developer', 'content': 'You are a model that can do function calling.'},
               {'role': 'user',      'content': ex['query']}]
    p_ids   = fg_proc.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                          return_dict=True, return_tensors='pt')['input_ids'][0]
    t_ids   = fg_proc(target, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    fg_train.append({'input_ids': torch.cat([p_ids, t_ids]),
                     'labels':    torch.cat([torch.full((len(p_ids),), -100, dtype=torch.long), t_ids])})

def collate(batch):
    L = max(len(x['input_ids']) for x in batch)
    inp = torch.zeros(len(batch), L, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    msk = torch.zeros(len(batch), L, dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x['input_ids'])
        inp[i, :n] = x['input_ids']
        lab[i, :n] = x['labels']
        msk[i, :n] = 1
    return {'input_ids': inp.to(DEVICE), 'attention_mask': msk.to(DEVICE), 'labels': lab.to(DEVICE)}

fg_model.train()
opt = AdamW(fg_model.parameters(), lr=FG_LR)
t0  = time.time()
for epoch in range(1, FG_EPOCHS + 1):
    random.shuffle(fg_train)
    ep_loss, n_b = 0.0, 0
    for i in range(0, len(fg_train), FG_BATCH):
        batch = collate(fg_train[i:i + FG_BATCH])
        loss  = fg_model(**batch).loss
        if loss is not None and not loss.isnan():
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fg_model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
            ep_loss += loss.item(); n_b += 1
    if epoch == 1 or epoch % 3 == 0:
        print(f'  Epoch {epoch:2d}/{FG_EPOCHS} loss={ep_loss/max(n_b,1):.4f}')
fg_train_s = time.time() - t0
print(f'  Done in {fg_train_s:.1f}s')
os.makedirs(FG_LORA, exist_ok=True)
fg_model.save_pretrained(FG_LORA)
fg_proc.save_pretrained(FG_LORA)
print(f'  Saved LoRA adapter → {FG_LORA}/')

print(f'\n[4/4] Evaluating on {len(test_data)} held-out queries...')
fg_model.eval()

m = {'nr': {'fn': 0, 'ex': 0, 'ms': []}, 'fg': {'fn': 0, 'ex': 0, 'ms': []}}

for i, tc in enumerate(test_data):
    sys.stdout.write(f'  {i+1}/{len(test_data)}\r'); sys.stdout.flush()
    q     = tc['query']
    label = tc['label']

    nr_r, nr_ms = router.route_query(nr_model, nr_tok, fn_vecs, nr_functions, q)
    m['nr']['ms'].append(nr_ms)
    if nr_r.get('function') == label.get('function'):
        m['nr']['fn'] += 1
    nr_clean = {k: v for k, v in nr_r.items() if v is not None}
    if nr_clean == label:
        m['nr']['ex'] += 1

    msgs = [{'role': 'developer', 'content': 'You are a model that can do function calling.'},
            {'role': 'user',      'content': q}]
    inp = fg_proc.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                      return_dict=True, return_tensors='pt')
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    t_fg = time.perf_counter()
    with torch.no_grad():
        out = fg_model.generate(**inp, pad_token_id=fg_proc.pad_token_id or 0,
                                max_new_tokens=60, do_sample=False)
    fg_ms = (time.perf_counter() - t_fg) * 1000
    m['fg']['ms'].append(fg_ms)
    raw  = fg_proc.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
    fg_r = parse_fg_output(raw) or {}
    if fg_r.get('function') == label.get('function'):
        m['fg']['fn'] += 1
    if fg_r == label:
        m['fg']['ex'] += 1

n = len(test_data)
W = 64
print(f'\n{"=" * W}')
print(f'  RESULTS   ({len(train_data)}-query train | {n}-query Ollama test)')
print(f'{"=" * W}')
print(f'  {"Model":<30} {"Fn%":>6} {"Exact%":>7} {"ms/q":>8} {"Train":>7} {"Size":>8}')
print(f'  {"-" * (W - 2)}')

def avg_ms(key): return sum(m[key]['ms']) / max(len(m[key]['ms']), 1)

nr_size_mb = nr_params * 4 / 1e6
fg_size_mb = fg_params * 2 / 1e6

print(f'  {"NanoRouter  (MiniLM-L6)":<30}'
      f' {m["nr"]["fn"]/n*100:5.1f}%  {m["nr"]["ex"]/n*100:6.1f}%  '
      f'{avg_ms("nr"):6.1f}ms  {nr_train_s:5.0f}s  {nr_size_mb:5.0f} MB')
print(f'  {"FuncGemma-270M (LoRA r=8)":<30}'
      f' {m["fg"]["fn"]/n*100:5.1f}%  {m["fg"]["ex"]/n*100:6.1f}%  '
      f'{avg_ms("fg"):6.1f}ms  {fg_train_s:5.0f}s  {fg_size_mb:5.0f} MB')
print(f'{"=" * W}')

speedup_lat   = avg_ms('fg') / max(avg_ms('nr'), 0.1)
speedup_train = fg_train_s / max(nr_train_s, 0.1)
speedup_size  = fg_size_mb / max(nr_size_mb, 0.1)
print(f'\n  Latency speedup:  {speedup_lat:.0f}×  ({avg_ms("fg"):.0f}ms → {avg_ms("nr"):.0f}ms)')
print(f'  Train   speedup:  {speedup_train:.0f}×  ({fg_train_s:.0f}s → {nr_train_s:.0f}s)')
print(f'  Size    ratio:    {speedup_size:.0f}×  ({fg_size_mb:.0f} MB → {nr_size_mb:.0f} MB)\n')
