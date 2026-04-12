import json, re, sys, time
from itertools import product
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
DEFAULT_GENERATOR = 'Qwen/Qwen2.5-0.5B-Instruct'
DEFAULT_N_PER_COMBO = 8
def load_schema(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)
def enumerate_combos(schema: dict) -> list[dict]:
    combos = []
    for fn in schema['functions']:
        params = fn.get('parameters', {})
        if not params:
            combos.append({'function': fn['name'], 'description': fn['description'], 'params': {}})
            continue
        param_names = list(params.keys())
        param_values = [params[p].get('enum', [params[p].get('example', '')]) for p in param_names]
        for values in product(*param_values):
            combos.append({'function': fn['name'], 'description': fn['description'], 'params': dict(zip(param_names, values))})
    return combos
def _make_seed(combo: dict) -> str:
    fn = combo['function'].replace('_', ' ').lower()
    params = combo['params']
    if not params:
        return fn
    target = list(params.values())[0]
    mapping = {'LIGHTS_ON': f'turn on the lights in the {target}', 'LIGHTS_OFF': f'turn off the lights in the {target}', 'LIGHTS_DIM': f'dim the lights in the {target}', 'LOCK': f'lock the {target}', 'UNLOCK': f'unlock the {target}', 'TEMP_SET': f'set the temperature in the {target}', 'BLINDS_UP': f'raise the blinds in the {target}', 'BLINDS_DOWN': f'lower the blinds in the {target}'}
    return mapping.get(combo['function'], f'{fn} {target}')
def build_prompt(combo: dict, n: int) -> str:
    seed = _make_seed(combo)
    fn = combo['function'].replace('_', ' ').lower()
    target = list(combo['params'].values())[0] if combo['params'] else ''
    forbidden = {w.lower() for w in (fn + ' ' + target).split() if len(w) > 2}
    forbidden_str = ', '.join(sorted(forbidden)[:8])
    return f'A user wants to: "{seed}" /no_think\nGenerate {n} realistic, casual ways a person might say this.\nRULES:\n- Avoid these obvious words: {forbidden_str}\n- Use slang, abbreviations, questions, indirect requests\n- Vary length: mix short commands and full sentences\n- Sound like real speech, not documentation\nReturn ONLY a JSON array of strings.\nOutput:'
def _strip_thinking(text: str) -> str:
    return re.sub('<think>.*?</think>', '', text, flags=re.DOTALL).strip()
def parse_utterances(text: str, n: int) -> list[str]:
    text = _strip_thinking(text)
    for m in re.finditer('\\[.*?\\]', text, re.DOTALL):
        try:
            items = json.loads(m.group())
            if isinstance(items, list) and all((isinstance(x, str) for x in items)):
                cleaned = [x.strip().strip('"') for x in items if isinstance(x, str) and 5 < len(x.strip()) < 100]
                if cleaned:
                    return cleaned[:n]
        except:
            pass
    items = re.findall('^\\d+[.)\\s]+(.{5,80})$', text, re.MULTILINE)
    if items:
        return [x.strip().strip('"') for x in items[:n]]
    items = re.findall('"([^"]{5,80})"', text)
    return [x.strip() for x in items if x.strip()][:n]
def _ollama_generate(prompt: str, model: str, host: str='http://localhost:11434') -> str:
    import urllib.request
    payload = json.dumps({'model': model, 'prompt': prompt, 'stream': False, 'think': False, 'options': {'temperature': 0.7, 'top_p': 0.9, 'num_predict': 256, 'num_ctx': 1024}}).encode()
    req = urllib.request.Request(f'{host}/api/generate', data=payload, headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp).get('response', '')
    except Exception as e:
        return ''
def generate_dataset(schema: dict, generator_model: str='qwen3.5:4b', n_per_combo: int=8, ollama_host: str='http://localhost:11434', verbose: bool=True, device: str='cuda') -> list[dict]:
    if verbose:
        print(f'  Generator  : {generator_model} via Ollama')
        print(f'  Strategy   : seed-based paraphrase (label always correct)')
        print(f'  N/combo    : {n_per_combo}')
    try:
        import urllib.request
        urllib.request.urlopen(f'{ollama_host}/api/tags', timeout=3)
    except Exception as e:
        raise RuntimeError(f'Ollama not reachable at {ollama_host}. Start with: ollama serve\nError: {e}')
    combos = enumerate_combos(schema)
    if verbose:
        print(f'  Combos     : {len(combos)} → {len(combos) * n_per_combo} examples target')
    examples = []
    t0 = time.time()
    for i, combo in enumerate(combos):
        seed = _make_seed(combo)
        prompt = build_prompt(combo, n_per_combo)
        raw = _ollama_generate(prompt, generator_model, ollama_host)
        utterances = parse_utterances(raw, n_per_combo)
        if not utterances:
            utterances = [seed]
        output_dict = {'function': combo['function'], **combo['params']}
        output_str = json.dumps(output_dict)
        for utt in utterances:
            if len(utt.split()) < 2:
                continue
            examples.append({'prompt': utt, 'output': output_str})
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            remaining = (len(combos) - i - 1) / max(i + 1, 1) * elapsed
            print(f'  [{i + 1:3d}/{len(combos)}] {len(examples):4d} examples | {elapsed:.0f}s | ~{remaining:.0f}s left')
    if verbose:
        print(f'\n  Generated {len(examples)} examples in {time.time() - t0:.1f}s')
        print(f'  Avg per combo: {len(examples) / max(len(combos), 1):.1f}')
    return examples
_ENRICHMENTS = {'blood pressure': ['BP', 'hypertension', 'antihypertensive'], 'diabetes': ['insulin', 'blood sugar', 'diabetic'], 'allergy': ['antihistamine', 'allergic', 'hay fever'], 'breakfast': ['morning meal', 'first meal', 'AM'], 'dinner': ['evening meal', 'supper', 'nighttime meal'], 'temperature': ['thermostat', 'temp', 'heat', 'degrees'], 'lights': ['lighting', 'lamps', 'bulbs'], 'lock': ['secure', 'deadbolt', 'bolt'], 'unlock': ['open', 'unsecure', 'access'], 'parking': ['spot', 'meter', 'space'], 'cancel': ['drop', 'remove', 'delete', 'axe'], 'schedule': ['book', 'set up', 'arrange', 'plan'], 'electronics': ['tech', 'gadgets', 'devices'], 'medication': ['meds', 'pills', 'prescription', 'Rx'], 'enroll': ['sign up', 'join', 'register', 'get into'], 'refill': ['reorder', 'running low', 'need more', 'resupply']}
def make_enriched_description(fn_name: str, fn_desc: str, params: dict) -> str:
    desc = f'{fn_name}: {fn_desc}'
    param_parts = []
    for k, v in params.items():
        extras = []
        v_lower = v.lower()
        for keyword, synonyms in _ENRICHMENTS.items():
            if keyword in v_lower or v_lower in keyword:
                extras.extend(synonyms[:3])
        if extras:
            param_parts.append(f"{k}={v} ({', '.join(extras)})")
        else:
            param_parts.append(f'{k}={v}')
    if param_parts:
        desc += '. ' + '; '.join(param_parts)
    return desc
def split_dataset(examples: list[dict], test_ratio: float=0.2, seed: int=42):
    import random
    random.seed(seed)
    random.shuffle(examples)
    n_test = max(int(len(examples) * test_ratio), 20)
    return (examples[n_test:], examples[:n_test])
def save_dataset(examples: list[dict], path: str, train: list=None, test: list=None):
    if train is not None and test is not None:
        data = {'train': train, 'test': test, 'total': len(train) + len(test)}
    else:
        train, test = split_dataset(examples)
        data = {'train': train, 'test': test, 'total': len(examples)}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data
if __name__ == '__main__':
    import sys, os
    sys.stdout.reconfigure(encoding='utf-8')
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    schema = load_schema('compact/examples/smart_home.json')
    combos = enumerate_combos(schema)
    print(f"Schema: {schema['name']}")
    print(f"Functions: {len(schema['functions'])}")
    print(f'Total (fn, params) combos: {len(combos)}')
    print(f'\nExample combos:')
    for c in combos[:4]:
        output = json.dumps({'function': c['function'], **c['params']})
        print(f"  {c['function']} | {c['params']} → {output}")
    print(f'\nGenerating prompt for first combo:')
    print(build_prompt(combos[0], n=5))