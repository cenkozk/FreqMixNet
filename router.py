"""
router.py — Two-stage schema-conditioned API router

Stage 1: Function routing — cosine similarity to fine-tuned function descriptions
Stage 2: Parameter extraction — per-param, per-type (enum/number/string/boolean)

Works for ANY schema: banking, restaurant, 3D game APIs, IoT device control, etc.
Supports enum, number, string, and boolean parameter types.

Usage:
  python router.py train --schema schema.json --out router.pt
  python router.py train --schema schema.json --out router.pt --examples data.json
  python router.py route --checkpoint router.pt --query "move north and attack with sword"
  python router.py eval  --checkpoint router.pt --test test.json

Schema format:
  {
    "name": "my_api",
    "functions": [
      {
        "name": "MOVE",
        "description": "Move the character in a direction",
        "parameters": {
          "direction": {"type": "enum", "values": ["north", "south", "east", "west"]},
          "speed":     {"type": "number", "description": "Movement speed multiplier"},
          "mode":      {"type": "string", "description": "Movement mode (walk/run/sneak)"},
          "sprint":    {"type": "boolean", "description": "Whether to sprint"}
        }
      }
    ]
  }
"""

import argparse
import itertools
import json
import re
import random
import sys
import time
import hashlib
import urllib.request
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MiniLMRouter(nn.Module):
    MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

    def __init__(self, num_bio_tags: int = 1):
        super().__init__()
        self.backbone    = AutoModel.from_pretrained(self.MODEL_NAME)
        self.temperature = nn.Parameter(torch.tensor(14.0))
        self.ner_head    = nn.Linear(384, num_bio_tags)

    @staticmethod
    def get_tokenizer():
        return AutoTokenizer.from_pretrained(MiniLMRouter.MODEL_NAME)

    def encode(self, input_ids, attention_mask):
        """Returns pooled, normalised embedding for function routing."""
        out  = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        vec  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(vec, dim=-1)

    def encode_tokens(self, input_ids, attention_mask):
        """Returns (token_embeddings, bio_logits) for MaxSim + slot filling."""
        out  = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        toks = out.last_hidden_state
        bio  = self.ner_head(toks)
        return toks, bio

    def score(self, q_vecs, t_vecs):
        return (q_vecs @ t_vecs.T) * self.temperature


def build_bio_tags(schema: dict) -> list[str]:
    """Derive BIO tag set from schema parameter names. O + B/I per param."""
    names: set[str] = set()
    for fn in schema.get('functions', []):
        for pname in fn.get('parameters', {}).keys():
            names.add(pname)
    tags = ['O']
    for p in sorted(names):
        tags.append(f'B-{p}')
        tags.append(f'I-{p}')
    return tags


def annotate_bio(text: str, slots: dict, tok, bio_tags: list[str],
                max_length: int = 64) -> list[int]:
    """
    Convert {param: value} slot dict to per-token BIO integer labels.
    Uses subword-level token matching — robust to any tokeniser vocabulary.
    Returns a list of length max_length (padded with O=0).
    """
    enc    = tok([text], max_length=max_length, truncation=True,
                 padding='max_length', return_tensors='pt')
    ids    = enc['input_ids'][0].tolist()
    qtoks  = tok.convert_ids_to_tokens(ids)
    labels = [0] * len(ids)

    for pname, value in slots.items():
        if value is None:
            continue
        b_tag = f'B-{pname}'
        i_tag = f'I-{pname}'
        if b_tag not in bio_tags:
            continue
        b_idx = bio_tags.index(b_tag)
        i_idx = bio_tags.index(i_tag) if i_tag in bio_tags else b_idx

        v_str    = str(value).replace('_', ' ').lower()
        val_ids  = tok.encode(v_str, add_special_tokens=False)
        val_toks = tok.convert_ids_to_tokens(val_ids)
        if not val_toks:
            continue

        body = qtoks[1:]
        for i in range(len(body) - len(val_toks) + 1):
            if body[i:i + len(val_toks)] == val_toks:
                pos = i + 1
                labels[pos] = b_idx
                for j in range(1, len(val_toks)):
                    if pos + j < len(labels):
                        labels[pos + j] = i_idx
                break

    return labels



_VERB_CLUSTERS: dict[str, list[str]] = {
    'send':     ['send', 'shoot', 'fire off', 'dispatch', 'forward', 'push'],
    'create':   ['create', 'make', 'set up', 'add', 'build', 'start', 'open'],
    'get':      ['get', 'fetch', 'retrieve', 'show', 'display', 'check', 'view', 'pull up', 'look up'],
    'cancel':   ['cancel', 'stop', 'remove', 'delete', 'drop', 'abort', 'undo', 'kill'],
    'update':   ['update', 'change', 'modify', 'edit', 'adjust', 'switch', 'revise', 'set'],
    'find':     ['find', 'search for', 'look for', 'locate', 'discover'],
    'book':     ['book', 'reserve', 'schedule', 'arrange', 'plan'],
    'pay':      ['pay', 'transfer', 'wire', 'move', 'send'],
    'navigate': ['navigate to', 'go to', 'take me to', 'head to', 'route to', 'get directions to'],
    'play':     ['play', 'start', 'launch', 'run', 'begin'],
    'cast':     ['cast', 'use', 'invoke', 'trigger', 'activate', 'fire'],
    'equip':    ['equip', 'wear', 'put on', 'switch to', 'use', 'wield'],
    'move':     ['move', 'go', 'walk', 'head', 'sneak', 'run'],
    'use':      ['use', 'consume', 'take', 'drink', 'apply'],
    'enable':   ['enable', 'turn on', 'activate', 'switch on', 'start'],
    'disable':  ['disable', 'turn off', 'deactivate', 'switch off', 'stop'],
    'default':  ['do', 'execute', 'perform', 'run', 'start', 'trigger'],
}

_AUG_TEMPLATES = [
    "{verb} {object}",
    "please {verb} {object}",
    "can you {verb} {object}",
    "I want to {verb} {object}",
    "I need to {verb} {object}",
    "I'd like to {verb} {object}",
    "help me {verb} {object}",
    "{verb} {object} now",
    "{verb} {object} please",
    "could you {verb} {object}",
    "go ahead and {verb} {object}",
    "{object}",
    "let's {verb} {object}",
    "quickly {verb} {object}",
    "I'm trying to {verb} {object}",
    "asap {verb} {object}",
    "just {verb} {object}",
    "hey, {verb} {object}",
    "{verb} the {object}",
    "want to {verb} {object}",
]


def _parse_desc(description: str) -> tuple[str, str]:
    """Extract (root_verb, object_phrase) from a function description string."""
    desc = description.strip().rstrip('.')
    words = desc.split()
    if not words:
        return 'do', desc.lower()
    verb = words[0].lower()
    rest = words[1:]
    while rest and rest[0].lower() in ('a', 'an', 'the'):
        rest = rest[1:]
    obj = ' '.join(rest).lower() if rest else desc.lower()
    return verb, obj


def _find_cluster(verb: str) -> list[str]:
    """Map a raw verb to its synonym cluster."""
    v = verb.lower()
    if v in _VERB_CLUSTERS:
        return _VERB_CLUSTERS[v]
    for key, syns in _VERB_CLUSTERS.items():
        if key in v or v in key:
            return syns
    return _VERB_CLUSTERS['default']


_ON_VERBS  = ['turn on', 'switch on', 'start', 'activate', 'enable',
              'put on', 'fire up', 'power on', 'wake up',
              'open', 'launch', 'boot up', 'power up']
_OFF_VERBS = ['turn off', 'switch off', 'shut down', 'disable', 'stop',
              'kill', 'deactivate', 'cut', 'close', 'shut off', 'power off',
              'shut', 'end']


def _slot_phrasings(fn_desc: str, slots: dict) -> list[str]:
    """
    Generate natural language phrasings for a given slot combination.
    Schema-agnostic: detects role of each param from name/value heuristics.
    Guaranteed: slots are ground-truth, phrasings reflect them.
    """
    state_val  = None
    state_key  = None
    loc_parts  = []
    obj_parts  = []

    for pname, val in slots.items():
        vstr = str(val).replace('_', ' ')
        vlo  = vstr.lower()
        plo  = pname.lower()
        if vlo in ('on', 'off', 'true', 'false', 'yes', 'no'):
            state_val = vlo
            state_key = pname
        elif any(w in plo for w in ('room', 'location', 'zone', 'area', 'place')):
            loc_parts.append(vstr)
        else:
            obj_parts.append(vstr)

    obj_str = ' '.join(obj_parts)
    loc_str = ' '.join(loc_parts)
    phrasings: list[str] = []

    if state_val in ('on', 'true', 'yes'):
        verbs = _ON_VERBS
    elif state_val in ('off', 'false', 'no'):
        verbs = _OFF_VERBS
    else:
        verbs = None

    if verbs:
        for v in verbs[:5]:
            if obj_str and loc_str:
                phrasings += [
                    f"{v} the {obj_str} in the {loc_str}",
                    f"{v} {loc_str} {obj_str}",
                    f"can you {v} the {obj_str}? it's in the {loc_str}",
                ]
            elif obj_str:
                phrasings += [f"{v} the {obj_str}", f"please {v} {obj_str}"]
            elif loc_str:
                phrasings.append(f"{v} everything in {loc_str}")
            else:
                phrasings.append(f"{v} it")
    else:
        verb0, _ = _parse_desc(fn_desc)
        syns = _find_cluster(verb0)
        for v in syns[:4]:
            if obj_str and loc_str:
                phrasings += [f"{v} {obj_str} in {loc_str}", f"{v} the {obj_str} in {loc_str}"]
            elif obj_str:
                phrasings.append(f"{v} {obj_str}")
            else:
                phrasings.append(f"{v} {loc_str}")

    seen: set[str] = set()
    result: list[str] = []
    for p in phrasings:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _programmatic_augment(fn: dict, examples_per_fn: int) -> list[dict]:
    """
    Generate (query, slots) pairs with GUARANTEED correct labels.

    Strategy:
    - Enumerate all combinations of enum param values.
    - For each combination, generate phrasings via _slot_phrasings().
    - Slots dict is the combination itself — no inference, no errors.
    - Also generate bare queries with only a subset of params to teach
      the model that not all params are always explicitly mentioned.
    """
    from itertools import product as iproduct
    desc   = fn.get('description', fn['name'].replace('_', ' ').title())
    params = fn.get('parameters', {})

    enum_params = {
        pname: (pdef.get('values') or pdef.get('enum'))
        for pname, pdef in params.items()
        if pdef.get('values') or pdef.get('enum')
    }

    if not enum_params:
        _NUM_SAMPLES  = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120]
        _STR_SAMPLES  = ['pasta', 'chicken', 'workout', 'egg', 'coffee', 'meeting']
        _NUM_TEMPLATES = [
            'set a timer for {n} minutes',
            'start a {n} minute timer',
            '{n} minute timer please',
            'remind me in {n} minutes',
            'countdown {n} minutes',
            '{n} min timer',
            'give me {n} minutes',
            'timer {n} minutes',
        ]
        _STR_TEMPLATES = [
            'set a {s} timer',
            '{n} minute {s} timer',
        ]
        results: list[dict] = []
        num_pnames = [p for p, d in params.items() if d.get('type') == 'number']
        str_pnames = [p for p, d in params.items() if d.get('type') == 'string']
        for n in _NUM_SAMPLES:
            for tpl in _NUM_TEMPLATES:
                results.append({'query': tpl.format(n=n), 'slots': {}})
        for s in _STR_SAMPLES:
            for n in _NUM_SAMPLES[:6]:
                for tpl in _STR_TEMPLATES:
                    results.append({'query': tpl.format(s=s, n=n), 'slots': {}})
        random.shuffle(results)
        return results[:examples_per_fn]

    pnames = list(enum_params.keys())
    pvals  = [enum_params[p] for p in pnames]

    all_combos = list(iproduct(*pvals))
    random.shuffle(all_combos)

    examples: list[dict] = []
    phrasings_per_combo = max(2, examples_per_fn // max(len(all_combos), 1))

    for combo in all_combos:
        slots = {pnames[i]: v for i, v in enumerate(combo)}
        phrases = _slot_phrasings(desc, slots)
        for _ in range(min(2, phrasings_per_combo)):
            sub_slots = {k: v for k, v in slots.items() if random.random() > 0.4}
            if sub_slots and sub_slots != slots:
                sub_phrases = _slot_phrasings(desc, sub_slots)
                if sub_phrases:
                    examples.append({
                        'query': random.choice(sub_phrases),
                        'slots': sub_slots,
                    })
        for phrase in phrases[:phrasings_per_combo]:
            examples.append({'query': phrase, 'slots': slots})

    random.shuffle(examples)
    return examples[:examples_per_fn]


def _call_openai(prompt: str) -> str:
    """Call OpenAI Chat API if OPENAI_API_KEY is set."""
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        return ''
    try:
        payload = json.dumps({
            'model': 'gpt-4o-mini',
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.8,
            'max_tokens': 600,
        }).encode()
        req = urllib.request.Request(
            'https://api.openai.com/v1/chat/completions', data=payload,
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {api_key}'}, method='POST'
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
            return data['choices'][0]['message']['content']
    except Exception as e:
        print(f'  [OpenAI] Failed: {e}')
        return ''


OLLAMA_BASE  = 'http://127.0.0.1:11434'
OLLAMA_MODEL = 'qwen3.5:2b'

def _call_ollama(prompt: str, timeout: int = 120) -> str:
    """Call Ollama /api/generate. think=false disables Qwen3 reasoning. 127.0.0.1 avoids IPv6."""
    try:
        payload = json.dumps({
            'model':      OLLAMA_MODEL,
            'prompt':     prompt,
            'stream':     False,
            'think':      False,
            'keep_alive': -1,
        }).encode()
        req = urllib.request.Request(
            f'{OLLAMA_BASE}/api/generate', data=payload,
            headers={'Content-Type': 'application/json'}, method='POST'
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
            return data.get('response', '')
    except Exception as e:
        print(f'Ollama error: {e}')
        return ''


def _warmup_ollama():
    """Preload model into VRAM once so generation calls need no cold-start."""
    print(f'  [Ollama] Loading {OLLAMA_MODEL} into VRAM...')
    result = _call_ollama('Say: ready', timeout=300)
    if result:
        print('  [Ollama] Model ready.')
    else:
        print('  [Error] Ollama warmup failed.')
        sys.exit(1)


def _call_llm(prompt: str) -> str:
    """Try OpenAI, then Ollama. Return empty string if both unavailable."""
    out = _call_openai(prompt)
    if out:
        return out
    return _call_ollama(prompt)


BATCH_SIZE = 5

def _llm_labeled_examples(fn: dict, examples_per_fn: int) -> list[dict] | None:
    """
    Ask an LLM to generate {query, slots} pairs.
    Returns list of dicts or None if unavailable.
    The LLM must output structured JSON so we get CORRECT slot labels,
    including implicit ones like 'kill the fan' → slots:{state:off}.
    """
    name   = fn['name']
    desc   = fn.get('description', name.replace('_', ' ').title())
    param_lines: list[str] = []
    for pname, pdef in fn.get('parameters', {}).items():
        vals = pdef.get('values') or pdef.get('enum')
        ptype = pdef.get('type', 'any')
        if vals:
            param_lines.append(f'  {pname} (enum): {vals}')
        else:
            param_lines.append(f'  {pname} ({ptype})')
    params_str = '\n'.join(param_lines) or '  (none)'

    results: list[dict] = []
    while len(results) < examples_per_fn:
        need   = min(BATCH_SIZE, examples_per_fn - len(results))
        prompt = (
            f"Generate exactly {need} diverse natural language queries for this API function.\n"
            f"Function: {name}\nDescription: {desc}\nParameters:\n{params_str}\n\n"
            f"For each query output a JSON object with two keys:\n"
            f'  "query": natural language command or question\n'
            f'  "slots": dict of which parameter values the query explicitly OR implicitly references\n'
            f"            (e.g. \'kill the fan\' → state:\"off\"; \'start dishwasher\' → state:\"on\")\n\n"
            f"Rules:\n"
            f"- Use varied phrasing (casual/formal/terse/typos).\n"
            f"- Cover different enum value combinations.\n"
            f"- Include implicit slot cases (synonyms that imply a value).\n"
            f"- Slots must only contain values from the enum lists above.\n\n"
            f'Output ONLY a raw JSON array: [{{"query":"...","slots":{{...}}}}, ...]'
        )
        response = _call_llm(prompt)
        if not response:
            break
        try:
            s = response.find('[')
            e = response.rfind(']') + 1
            if s != -1 and e > s:
                batch = json.loads(response[s:e])
                for item in batch:
                    if isinstance(item, dict) and 'query' in item:
                        q = str(item['query']).strip()
                        slots = {k: str(v) for k, v in item.get('slots', {}).items()}
                        if q and not q.startswith('{') and len(q) >= 4:
                            results.append({'query': q, 'slots': slots})
        except Exception:
            pass

    return results if results else None



def generate_data(schema: dict, examples_per_fn: int = 100) -> list[dict]:
    """
    Generate training examples from a schema.

    4-tier pipeline:
      1. Disk cache  — instant.
      2. OpenAI API  — best quality, structured {query, slots} output.
      3. Ollama      — local LLM, structured {query, slots} output.
      4. Programmatic augmentation — deterministic, zero external deps;
                                     always produces correct slot labels.

    Each returned example: {'query': str, 'function': str,
                            'params': dict, 'slots': dict}
    """
    schema_str  = json.dumps(schema, sort_keys=True)
    schema_hash = hashlib.md5(schema_str.encode()).hexdigest()[:8]
    cache_path  = f'.cache_synth_{schema_hash}.json'

    if os.path.exists(cache_path):
        print(f'  [Data] Cache hit (hash: {schema_hash})')
        with open(cache_path) as f:
            return json.load(f)

    llm_available = bool(os.environ.get('OPENAI_API_KEY'))
    if not llm_available:
        print('  [Data] Checking Ollama...')
        try:
            req = urllib.request.Request(f'{OLLAMA_BASE}/api/tags', method='GET')
            with urllib.request.urlopen(req, timeout=5) as r:
                json.load(r)
            llm_available = True
        except Exception:
            pass

    if llm_available:
        tier = 'OpenAI' if os.environ.get('OPENAI_API_KEY') else 'Ollama'
        print(f'  [Data] Generating {examples_per_fn} examples/fn via {tier}...')
        if tier == 'Ollama':
            _warmup_ollama()
    else:
        print('  [Data] No LLM available — using programmatic augmentation.')

    examples: list[dict] = []

    for fn in schema['functions']:
        name = fn['name']
        labeled: list[dict] | None = None

        if llm_available:
            labeled = _llm_labeled_examples(fn, examples_per_fn)
            if not labeled:
                print(f'  [Warning] LLM failed for {name} — using programmatic augmentation')

        if not labeled:
            labeled = _programmatic_augment(fn, examples_per_fn)

        for item in labeled:
            q     = str(item.get('query', '')).strip()
            slots = {k: str(v) for k, v in item.get('slots', {}).items()}
            if not q or q.startswith('{') or q.startswith('[') or len(q) < 4:
                continue
            examples.append({
                'query':    q,
                'function': name,
                'params':   slots,
                'slots':    slots,
            })

    random.shuffle(examples)
    try:
        with open(cache_path, 'w') as f:
            json.dump(examples, f, indent=2)
    except Exception:
        pass
    return examples



@torch.no_grad()
def _maxsim(q_toks: torch.Tensor, v_toks: torch.Tensor) -> float:
    """
    MaxSim in V→Q direction: mean over VALUE tokens of max cosine-dot against
    any query token. Asks 'how much of this value string is present in the query?'
    This is the correct direction for enum selection — a value token like 'tv'
    that appears literally in the query scores very high; 'dishwasher' tokens
    that don't appear in the query score low. Avoids length bias of Q→V.
    q_toks: (q_len, 384)   v_toks: (v_len, 384)
    """
    q_norm = F.normalize(q_toks, dim=-1)
    v_norm = F.normalize(v_toks, dim=-1)
    scores = v_norm @ q_norm.T
    body = scores[1:-1] if scores.shape[0] > 2 else scores
    return body.max(dim=-1).values.mean().item()


@torch.no_grad()
def _pick_enum(model, tok, query: str, fn_desc: str, param_name: str,
               param_def: dict) -> str:
    """
    Two-tier enum selection:
    1. Literal word match  — if an enum value (or its alias) appears verbatim in
       the query, return it immediately (fastest, handles on/turn on/off/room names).
    2. Token-level MaxSim — for implicit/paraphrase cases (close→off, kill→off).
    """
    values = param_def.get('values', [])
    if not values:
        return None
    if len(values) == 1:
        return values[0]

    ql = query.lower()

    for v in values:
        alias = str(v).replace('_', ' ').lower()
        if re.search(rf'\b{re.escape(alias)}\b', ql):
            return v

    q_enc  = tok([query], padding=False, truncation=True,
                 max_length=64, return_tensors='pt').to(DEVICE)
    q_toks, _ = model.encode_tokens(q_enc.input_ids, q_enc.attention_mask)
    q_toks = q_toks[0]

    best_val, best_score = values[0], -1.0
    for v in values:
        anchor = f"{param_name}: {str(v).replace('_', ' ')}"
        v_enc  = tok([anchor], padding=False, truncation=True,
                     max_length=16, return_tensors='pt').to(DEVICE)
        v_toks, _ = model.encode_tokens(v_enc.input_ids, v_enc.attention_mask)
        score = _maxsim(q_toks, v_toks[0])
        if score > best_score:
            best_score = score
            best_val   = v

    return best_val


def _extract_number(query: str, param_name: str, param_def: dict):
    """Extract first number from query. Returns int if whole, float otherwise."""
    numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', query)
    if numbers:
        n = numbers[0].replace(',', '.')
        return float(n) if '.' in n else int(n)
    return param_def.get('default', None)


def _extract_boolean(query: str, param_name: str, param_def: dict) -> bool:
    """Detect affirmative/negative intent for boolean parameters."""
    q = query.lower()
    negatives = {'no', "don't", 'not', 'without', 'disable', 'off', 'false', 'never'}
    if any(w in q.split() for w in negatives):
        return False
    return True


def _extract_string(query: str, param_name: str, param_def: dict,
                    model=None, tok=None):
    """
    3-tier semantic span extractor for open-ended string parameters.

    Tier 1 — Quoted passthrough (highest confidence).
    Tier 2 — Dual-anchor n-gram scoring via MiniLM:
        Scores all contiguous spans against both the param description and
        param name, averages the two similarity scores, then applies a
        length penalty to prefer concise entity spans over long verb phrases.
    Tier 3 — Regex heuristic fallback.
    """
    quoted = re.findall(r'"([^"]+)"|\x27([^\x27]+)\x27', query)
    if quoted:
        return (quoted[0][0] or quoted[0][1]).strip()

    words = query.split()
    if not words:
        return param_def.get('default', None)

    if model is not None and tok is not None:
        STOP = {
            'i', 'a', 'an', 'the', 'to', 'of', 'in', 'is', 'it', 'me', 'my',
            'we', 'he', 'she', 'they', 'and', 'or', 'but', 'with', 'for', 'on',
            'at', 'by', 'from', 'that', 'this', 'be', 'am', 'are', 'was', 'were',
            'can', 'could', 'will', 'would', 'please', 'send', 'tell', 'write',
            'set', 'create', 'make', 'add', 'get', 'do', 'new', 'just', 'let',
        }
        candidates = []
        for n in range(1, min(6, len(words) + 1)):
            for i in range(len(words) - n + 1):
                span = ' '.join(words[i:i + n])
                if not set(span.lower().split()).issubset(STOP):
                    candidates.append(span)

        if not candidates:
            candidates = [query]

        param_label = param_def.get('description', param_name.replace('_', ' '))
        anchors = [param_label, param_name.replace('_', ' ')]

        with torch.no_grad():
            a_enc  = tok(anchors,    padding=True, truncation=True, max_length=32, return_tensors='pt').to(DEVICE)
            c_enc  = tok(candidates, padding=True, truncation=True, max_length=32, return_tensors='pt').to(DEVICE)
            a_vecs = model.encode(a_enc.input_ids, a_enc.attention_mask)
            c_vecs = model.encode(c_enc.input_ids, c_enc.attention_mask)

        raw_scores     = (c_vecs @ a_vecs.T).mean(dim=1)
        length_penalty = torch.tensor(
            [0.05 * len(c.split()) for c in candidates], device=DEVICE
        )
        scores   = raw_scores - length_penalty
        best_idx = int(scores.argmax().item())
        return candidates[best_idx]

    heuristics = [
        r'(?:message|say|tell|send|write|note|comment|description)[:\s]+[\'"]*(.+?)[\'"]*$',
        r'(?:named?|called?|titled?|labeled?)\s+[\'"]?([^\'"]+)[\'"]?',
        r'(?:with text|containing|about)\s+[\'"]?(.+)[\'"]?',
    ]
    for pattern in heuristics:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return param_def.get('default', None)


def extract_params(model, tok, query: str, fn_def: dict) -> dict:
    """
    Stage 2: extract all parameters for a given function from the query.

    Supports parameter definitions in two formats:
      JSON Schema:  {"type": "string", "enum": ["hotel", "flight"]}
      Custom:       {"type": "enum",   "values": ["hotel", "flight"]}
    """
    params = {}
    fn_desc = fn_def.get('description', fn_def['name'])
    for param_name, param_def in fn_def.get('parameters', {}).items():
        ptype       = param_def.get('type', 'string')
        enum_values = param_def.get('values') or param_def.get('enum')

        if enum_values:
            params[param_name] = _pick_enum(model, tok, query, fn_desc,
                                            param_name,
                                            {**param_def, 'values': enum_values})
        elif ptype in ('number', 'integer'):
            params[param_name] = _extract_number(query, param_name, param_def)
        elif ptype == 'boolean':
            params[param_name] = _extract_boolean(query, param_name, param_def)
        else:
            params[param_name] = _extract_string(query, param_name, param_def,
                                                  model=model, tok=tok)
    return params


def _cat_summary(schema: dict) -> str:
    return "dynamic (LLM synthesis)"


def _train_model(schema: dict, epochs: int, lr: float, batch_size: int,
                 extra_examples: list = None,
                 examples_only: bool = False) -> tuple:
    """
    Train the MiniLM bi-encoder on a schema.

    Args:
        examples_only: if True, skip the internal generate_data() call and
                       train ONLY on extra_examples. Use this in benchmarks
                       to avoid data leakage from the cached synthetic set.
    """
    bio_tags  = build_bio_tags(schema)
    tok       = MiniLMRouter.get_tokenizer()
    model     = MiniLMRouter(num_bio_tags=len(bio_tags)).to(DEVICE)
    functions = schema['functions']

    if examples_only and extra_examples:
        examples = []
        for ex in extra_examples:
            label   = json.loads(ex['output']) if isinstance(ex.get('output'), str) \
                      else ex.get('label', {})
            fn_name = label.get('function', ex.get('function', ''))
            params  = {k: v for k, v in label.items() if k != 'function'} \
                      if 'function' in label else ex.get('params', {})
            examples.append({
                'query':    ex['prompt'] if 'prompt' in ex else ex.get('query', ''),
                'function': fn_name,
                'params':   params,
            })
        print(f'  Training on {len(examples)} provided examples only (no synthetic generation)')
    else:
        examples = generate_data(schema)
        if extra_examples:
            for ex in extra_examples:
                label   = json.loads(ex['output']) if isinstance(ex.get('output'), str) \
                          else ex.get('label', {})
                fn_name = label.get('function', '')
                params  = {k: v for k, v in label.items() if k != 'function'}
                examples.append({
                    'query':    ex['prompt'] if 'prompt' in ex else ex.get('query', ''),
                    'function': fn_name,
                    'params':   params,
                })
            print(f'  + {len(extra_examples)} user examples mixed in (with param labels)')

    random.shuffle(examples)
    print(f'  Generated {len(examples)} training examples for {len(functions)} functions')
    print(f'  Param types: {_param_type_summary(schema)}')
    print(f'  Category breakdown: {_cat_summary(schema)}')

    fn_descriptions = [fn.get('description', fn['name']) for fn in functions]
    fn_to_idx       = {fn['name']: i for i, fn in enumerate(functions)}
    t_enc = tok(fn_descriptions, padding=True, truncation=True,
                max_length=64, return_tensors='pt').to(DEVICE)

    param_val_toks = {}
    with torch.no_grad():
        for fn in functions:
            fn_dict = {}
            for pname, pdef in fn.get('parameters', {}).items():
                vals = pdef.get('values') or pdef.get('enum') or []
                if len(vals) >= 2:
                    val_dict = {}
                    for v in vals:
                        anchor = f"{pname}: {str(v).replace('_', ' ')}"
                        a_enc = tok([anchor], padding=False, truncation=True,
                                    max_length=16, return_tensors='pt').to(DEVICE)
                        a_toks, _ = model.encode_tokens(a_enc.input_ids, a_enc.attention_mask)
                        val_dict[v] = F.normalize(a_toks[0], dim=-1)
                    fn_dict[pname] = (vals, val_dict)
            param_val_toks[fn['name']] = fn_dict
    has_param_loss = any(param_val_toks.values())

    BIO_LAMBDA   = 0.3
    PARAM_LAMBDA = 0.5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    ce        = nn.CrossEntropyLoss()
    t0        = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(examples)
        total_loss, n_batches = 0.0, 0

        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            if not batch:
                continue

            queries        = [e['query'] for e in batch]
            target_indices = [fn_to_idx.get(e['function'], 0) for e in batch]

            q_enc  = tok(queries, padding=True, truncation=True,
                         max_length=64, return_tensors='pt').to(DEVICE)
            q_vecs = model.encode(q_enc.input_ids, q_enc.attention_mask)
            t_vecs = model.encode(t_enc.input_ids, t_enc.attention_mask)

            fn_scores = model.score(q_vecs, t_vecs)
            fn_labels = torch.tensor(target_indices, device=DEVICE)
            loss      = ce(fn_scores, fn_labels)

            need_toks = (has_param_loss and any(e.get('params') for e in batch)) or \
                        (len(bio_tags) > 1 and any(e.get('slots') for e in batch))

            if need_toks:
                q_toks_all, bio_logits_all = model.encode_tokens(q_enc.input_ids, q_enc.attention_mask)
            else:
                q_toks_all = bio_logits_all = None

            if has_param_loss and q_toks_all is not None:
                MARGIN = 0.3
                param_losses = []
                for i, ex in enumerate(batch):
                    fn_params = param_val_toks.get(ex.get('function', ''), {})
                    ex_params = ex.get('params', {})
                    if not fn_params or not ex_params:
                        continue
                    q_norm_i = F.normalize(q_toks_all[i], dim=-1)
                    for pname, correct_val in ex_params.items():
                        entry = fn_params.get(pname)
                        if entry is None:
                            continue
                        vals, val_toks_dict = entry
                        if correct_val not in vals:
                            continue
                        maxsim_scores = []
                        for v in vals:
                            a_norm = val_toks_dict[v]
                            sim    = a_norm[1:-1] @ q_norm_i.T
                            maxsim_scores.append(sim.max(dim=-1).values.mean())
                        scores_t    = torch.stack(maxsim_scores) * model.temperature.detach()
                        correct_idx = vals.index(correct_val)
                        p_label     = torch.tensor([correct_idx], device=DEVICE)
                        ce_loss     = ce(scores_t.unsqueeze(0), p_label)
                        hard_neg    = scores_t[[j for j in range(len(vals)) if j != correct_idx]].max()
                        triplet     = F.relu(hard_neg - scores_t[correct_idx] + MARGIN)
                        param_losses.append(ce_loss + triplet)

                if param_losses:
                    loss = loss + PARAM_LAMBDA * torch.stack(param_losses).mean()

            if bio_logits_all is not None and len(bio_tags) > 1 and any(e.get('slots') for e in batch):
                seq_len = bio_logits_all.shape[1]
                bio_label_list = []
                for ex in batch:
                    raw = annotate_bio(ex['query'], ex.get('params', {}),
                                      tok, bio_tags)
                    raw = (raw + [0] * seq_len)[:seq_len]
                    bio_label_list.append(raw)
                bio_labels = torch.tensor(bio_label_list, device=DEVICE)
                bio_loss   = ce(bio_logits_all.reshape(-1, len(bio_tags)),
                                bio_labels.reshape(-1))
                loss = loss + BIO_LAMBDA * bio_loss


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        if epoch == 1 or epoch % 10 == 0:
            acc = _quick_eval(model, tok, t_enc, examples[:200], fn_to_idx)
            print(f'  Epoch {epoch:3d}/{epochs} | loss={total_loss/max(n_batches,1):.3f}'
                  f' | temp={model.temperature.item():.2f} | fn_acc={acc*100:.1f}%'
                  f' | {time.time()-t0:.0f}s')

    return model, tok, functions, bio_tags





def _param_type_summary(schema: dict) -> str:
    types = {}
    for fn in schema['functions']:
        for pdef in fn.get('parameters', {}).values():
            if pdef.get('values') or pdef.get('enum'):
                t = 'enum'
            else:
                t = pdef.get('type', 'string')
            types[t] = types.get(t, 0) + 1
    return ', '.join(f'{v}x{k}' for k, v in sorted(types.items())) or 'none'


@torch.no_grad()
def _quick_eval(model, tok, t_enc, examples, fn_to_idx):
    model.eval()
    t_vecs  = model.encode(t_enc.input_ids, t_enc.attention_mask)
    correct = total = 0
    for e in examples:
        q_enc  = tok([e['query']], padding=True, truncation=True,
                     max_length=64, return_tensors='pt').to(DEVICE)
        q_vecs = model.encode(q_enc.input_ids, q_enc.attention_mask)
        pred   = model.score(q_vecs, t_vecs)[0].argmax().item()
        if pred == fn_to_idx.get(e['function'], -1):
            correct += 1
        total += 1
    model.train()
    return correct / max(total, 1)


@torch.no_grad()
def _build_fn_cache(model, tok, functions: list) -> torch.Tensor:
    """Pre-encode all function descriptions → (N_fns, 384)."""
    model.eval()
    fn_descriptions = [fn.get('description', fn['name']) for fn in functions]
    t_enc = tok(fn_descriptions, padding=True, truncation=True,
                max_length=64, return_tensors='pt').to(DEVICE)
    return model.encode(t_enc.input_ids, t_enc.attention_mask)


@torch.no_grad()
def route_query(model, tok, fn_vecs, functions: list, query: str) -> tuple[dict, float]:
    """
    Two-stage routing.
    Stage 1: cosine similarity → function
    Stage 2: per-param extraction → parameter values
    Returns (result dict, latency_ms).
    """
    model.eval()
    q_enc = tok([query], padding=True, truncation=True,
                max_length=64, return_tensors='pt').to(DEVICE)

    t0     = time.perf_counter()
    q_vecs = model.encode(q_enc.input_ids, q_enc.attention_mask)
    scores = model.score(q_vecs, fn_vecs)
    best_fn_idx = scores[0].argmax().item()
    fn_def = functions[best_fn_idx]

    params = extract_params(model, tok, query, fn_def)
    ms     = (time.perf_counter() - t0) * 1000

    return {'function': fn_def['name'], **params}, ms


def _load_model(ckpt_path: str):
    ckpt      = torch.load(ckpt_path, map_location=DEVICE)
    bio_tags  = ckpt.get('bio_tags', ['O'])
    tok       = MiniLMRouter.get_tokenizer()
    model     = MiniLMRouter(num_bio_tags=len(bio_tags)).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    functions = ckpt['functions']
    fn_vecs   = _build_fn_cache(model, tok, functions)
    return model, tok, fn_vecs, functions, bio_tags


def cmd_train(args):
    print(f'\n  Loading schema: {args.schema}')
    with open(args.schema) as f:
        schema = json.load(f)
    n_fns = len(schema['functions'])
    print(f'  Schema: {schema.get("name", "?")} | {n_fns} functions')
    print(f'  Encoder: {MiniLMRouter.MODEL_NAME}')

    extra: list = []
    if args.examples:
        for path in args.examples:
            with open(path) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                raw = raw.get('train', []) + raw.get('examples', [])
            extra.extend(raw)
        print(f'  User examples: {len(extra)} loaded from {len(args.examples)} file(s)')

    if args.no_cache:
        schema_str  = json.dumps(schema, sort_keys=True)
        schema_hash = hashlib.md5(schema_str.encode()).hexdigest()[:8]
        cache_path  = f'.cache_synth_{schema_hash}.json'
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f'  Cache cleared ({cache_path})')

    if extra:
        print('  Data source  : [1] user examples (Tier 0 — highest quality for niche domains)')
    elif os.environ.get('OPENAI_API_KEY'):
        print('  Data source  : [2] OpenAI API synthesis (set OPENAI_API_KEY)')
    else:
        print('  Data source  : [4] programmatic augmentation (zero deps)')
        print('  Tip: supply --examples my_examples.json for niche/domain-specific APIs')
        print('       or set OPENAI_API_KEY for automatic LLM synthesis')

    print('\n  Training...')
    model, tok, functions, bio_tags = _train_model(
        schema, args.epochs, args.lr, args.batch_size, extra, args.examples_only)

    torch.save({
        'model_state': model.state_dict(),
        'functions':   functions,
        'schema_name': schema.get('name', 'unknown'),
        'bio_tags':    bio_tags,
    }, args.out)
    print(f'\n  Saved -> {args.out}  ({len(functions)} functions, {len(bio_tags)} BIO tags)')


def cmd_route(args):
    model, tok, fn_vecs, functions, bio_tags = _load_model(args.checkpoint)
    query = args.query or input('Query: ')
    result, ms = route_query(model, tok, fn_vecs, functions, query)
    print(f'\n  Query:   {query}')
    print(f'  Routed -> {result}')
    print(f'  Latency: {ms:.1f} ms')


def cmd_eval(args):
    model, tok, fn_vecs, functions, bio_tags = _load_model(args.checkpoint)
    with open(args.test) as f:
        test_cases = json.load(f)

    fn_correct = ex_correct = total = 0
    latencies = []

    for tc in test_cases:
        pred, ms = route_query(model, tok, fn_vecs, functions, tc['query'])
        label    = tc['label']
        latencies.append(ms)
        if pred.get('function') == label.get('function'): fn_correct += 1
        if pred == label:                                  ex_correct += 1
        total += 1

    print(f'\n  Test cases:     {total}')
    print(f'  Fn accuracy:    {fn_correct/total*100:.1f}%')
    print(f'  Exact accuracy: {ex_correct/total*100:.1f}%')
    print(f'  Avg latency:    {sum(latencies)/len(latencies):.1f} ms')
    print(f'  p95 latency:    {sorted(latencies)[int(len(latencies)*0.95)]:.1f} ms')


def cmd_init(args):
    """
    Scaffold a starter examples.json for a schema.
    The user fills in diverse natural-language queries — this is the
    fastest way to get great accuracy on any niche or domain-specific API.
    """
    with open(args.schema) as f:
        schema = json.load(f)

    examples = []
    for fn in schema['functions']:
        name   = fn['name']
        desc   = fn.get('description', name.replace('_', ' '))
        params = {}
        for pname, pdef in fn.get('parameters', {}).items():
            vals = pdef.get('enum') or pdef.get('values') or []
            params[pname] = vals[0] if vals else f'<{pname}_value>'

        verb0, obj0 = _parse_desc(desc)
        syns = _find_cluster(verb0)[:3]
        for syn in syns:
            examples.append({
                'query':    f'{syn} {obj0}',
                'function': name,
                'params':   params,
            })

    out_path = args.out or f"{schema.get('name', 'schema')}_examples.json"
    with open(out_path, 'w') as f:
        json.dump(examples, f, indent=2)

    n = len(schema['functions'])
    print(f"\n  Scaffolded {len(examples)} example slots across {n} functions.")
    print(f"  Output: {out_path}")
    print(f"""\n  Next steps:
  1. Open {out_path} and replace each 'query' with a realistic user phrase.
     Aim for 5-10 diverse phrasings per function (short, long, casual, terse).
  2. Train with your examples:
     python router.py train --schema {args.schema} --examples {out_path} --out router.pt
  3. Route a query:
     python router.py route --checkpoint router.pt --query "your command here"
""")


def main():
    p   = argparse.ArgumentParser(
        description='NanoRouter — schema-aware API function router',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start:
  python router.py train --schema examples/banking.json --out bank.pt
  python router.py route --checkpoint bank.pt --query "transfer 50 to savings"

For niche/domain-specific APIs (best accuracy):
  python router.py init  --schema examples/drone.json          # scaffold examples file
  # edit drone_examples.json — fill in real user phrases
  python router.py train --schema examples/drone.json --examples drone_examples.json --out drone.pt

Data synthesis order (automatic, no config needed):
  1. Your --examples file  (Tier 0 — best, works for any domain)
  2. OpenAI API            (set OPENAI_API_KEY)
  3. Ollama local          (run: ollama serve + ollama pull qwen3:4b)
  4. Programmatic          (zero deps — covers common API verbs)
"""
    )
    sub = p.add_subparsers(dest='command', required=True)

    ini = sub.add_parser('init',
        help='Scaffold a starter examples.json for any schema (fill it in for niche domains)')
    ini.add_argument('--schema', required=True, help='Path to schema JSON')
    ini.add_argument('--out',    default='',
        help='Output path (default: <schema_name>_examples.json)')

    tr = sub.add_parser('train',
        help='Fine-tune the router for a schema (runs in ~30-150 seconds)')
    tr.add_argument('--schema',     required=True,
        help='Path to schema JSON (defines functions + parameters)')
    tr.add_argument('--out',        default='router.pt',
        help='Output checkpoint path (default: router.pt)')
    tr.add_argument('--examples',   nargs='+', default=[],
        help='One or more JSON files with {query, function, params} examples. '
             'Use this for niche domains where programmatic augmentation may be weak. '
             'Run `init` to scaffold a starter file.')
    tr.add_argument('--epochs',     type=int,   default=40,
        help='Training epochs (default: 40, increase for harder schemas)')
    tr.add_argument('--lr',         type=float, default=1e-4,
        help='Learning rate (default: 1e-4)')
    tr.add_argument('--batch-size', type=int,   default=32,   dest='batch_size')
    tr.add_argument('--no-cache',   action='store_true',      dest='no_cache',
        help='Force re-generation of synthetic training data (ignores disk cache)')
    tr.add_argument('--examples-only', action='store_true',   dest='examples_only',
        help='Skip synthesizing extra data and only use provided examples')

    ro = sub.add_parser('route',
        help='Route a single query using a trained checkpoint')
    ro.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    ro.add_argument('--query',      default='',
        help='Query string (omit to enter interactively)')

    ev = sub.add_parser('eval',
        help='Evaluate accuracy and latency on a labelled test set')
    ev.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    ev.add_argument('--test',       required=True,
        help='JSON file: [{"query": "...", "label": {"function": ..., ...}}]')

    args = p.parse_args()
    {'init': cmd_init, 'train': cmd_train, 'route': cmd_route, 'eval': cmd_eval}[args.command](args)


if __name__ == '__main__':
    main()
