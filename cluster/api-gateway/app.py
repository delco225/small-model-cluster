from fastapi import FastAPI, HTTPException, Request
import os, requests, uuid, time, resource
from typing import Dict, List, Deque
from collections import deque

app = FastAPI()

STORIES: Dict[str, Dict] = {}
DEFAULT_PARTS = 10
MAX_RECENT = 100
RECENT: Deque[Dict] = deque(maxlen=MAX_RECENT)

GEN_MAX_TOKENS_DEFAULT = int(os.getenv('GEN_MAX_TOKENS', '512'))
GEN_TEMP_DEFAULT = float(os.getenv('GEN_TEMPERATURE', '0.7'))

@app.middleware("http")
async def timing_mw(request: Request, call_next):
    wall_start = time.time()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    resp = None
    try:
        resp = await call_next(request)
        return resp
    finally:
        wall_ms = (time.time() - wall_start) * 1000.0
        usage_end = resource.getrusage(resource.RUSAGE_SELF)
        cpu_user_ms = (usage_end.ru_utime - usage_start.ru_utime) * 1000.0
        cpu_sys_ms = (usage_end.ru_stime - usage_start.ru_stime) * 1000.0
        entry = {
            'path': request.url.path,
            'query': str(request.url.query)[:200],
            'status': getattr(resp, 'status_code', None),
            'wall_ms': round(wall_ms, 2),
            'cpu_user_ms': round(cpu_user_ms, 2),
            'cpu_sys_ms': round(cpu_sys_ms, 2),
            'time': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
        }
        RECENT.append(entry)
        if resp is not None:
            resp.headers['X-Wall-Time-ms'] = str(entry['wall_ms'])
            resp.headers['X-CPU-User-ms'] = str(entry['cpu_user_ms'])
            resp.headers['X-CPU-Sys-ms'] = str(entry['cpu_sys_ms'])

# Updated helper: tries Mistral (OpenAI completions) then Phi3 (Ollama generate)
def call_model(prompt: str, prefer: str = None, max_tokens: int = None, temperature: float = None):
    if max_tokens is None:
        max_tokens = GEN_MAX_TOKENS_DEFAULT
    if temperature is None:
        temperature = GEN_TEMP_DEFAULT
    mistral = os.getenv('MODEL_MISTRAL_ENDPOINT')  # e.g. http://mistral-vllm:8000
    phi3 = os.getenv('MODEL_PHI3_ENDPOINT')        # e.g. http://ollama-amd:11434
    attempts = []
    def add_mistral():
        if mistral:
            attempts.append(('mistral', f"{mistral}/v1/completions", {
                'model': os.getenv('MISTRAL_MODEL', 'mistral-7b-instruct-v0.2'),
                'prompt': prompt,
                'max_tokens': max_tokens,
                'temperature': temperature
            }))
    def add_phi3():
        if phi3:
            attempts.append(('phi3', f"{phi3}/api/generate", {
                'model': os.getenv('PHI3_MODEL', 'phi3:mini'),
                'prompt': prompt,
                'options': {'temperature': temperature, 'num_predict': max_tokens}
            }))
    if prefer == 'mistral': add_mistral()
    if prefer == 'phi3': add_phi3()
    if prefer is None:
        add_mistral(); add_phi3()
    errors = []
    for name, url, payload in attempts:
        t0 = time.time()
        try:
            r = requests.post(url, json=payload, timeout=180)
            latency_ms = (time.time() - t0) * 1000.0
            if r.ok:
                data = r.json()
                if name == 'mistral' and 'choices' in data:
                    txt = ''.join(choice.get('text','') for choice in data['choices']).strip()
                    return {'model': name, 'text': txt, 'raw': data, 'latency_ms': round(latency_ms,2)}
                if name == 'phi3':
                    txt = data.get('response') or data.get('generated_text') or data.get('text') or ''
                    if not txt and isinstance(data, dict) and data.get('done') is False:
                        txt = str(data)
                    return {'model': name, 'text': txt.strip(), 'raw': data, 'latency_ms': round(latency_ms,2)}
                generic = data.get('generated_text') or data.get('response') or data.get('text') or str(data)
                return {'model': name, 'text': generic.strip(), 'raw': data, 'latency_ms': round(latency_ms,2)}
            else:
                errors.append({'model': name, 'status': r.status_code, 'body': r.text[:300]})
        except Exception as e:
            errors.append({'model': name, 'exception': str(e)})
    return {'error': 'No model succeeded', 'attempts': errors}

@app.get('/perf/recent')
def perf_recent():
    return list(RECENT)[-25:]

@app.get('/health')
def health():
    return {'status': 'ok', 'stories': len(STORIES), 'recent_requests': len(RECENT)}

# Allow both POST and GET for convenience when testing in a browser
@app.get('/story/init')
@app.post('/story/init')
def init_story(title: str, initial_paragraph: str, preferred_model: str = None, total_parts: int = DEFAULT_PARTS, max_tokens: int = None, temperature: float = None):
    if total_parts < 1 or total_parts > 50:
        raise HTTPException(status_code=400, detail='total_parts must be between 1 and 50')
    story_id = str(uuid.uuid4())
    prompt = ("You are an expert fiction author. Expand the following opening into Part 1 of a multi-part story. "
              f"Aim for vivid imagery, consistent POV, and a hook for continuation. Title: {title}\nOpening: {initial_paragraph}\nPart 1:")
    result = call_model(prompt, prefer=preferred_model, max_tokens=max_tokens, temperature=temperature)
    if 'error' in result:
        raise HTTPException(status_code=502, detail=result)
    STORIES[story_id] = {
        'title': title,
        'parts': [result['text'].strip()],
        'model': result['model'],
        'created': time.time(),
        'total_parts': total_parts
    }
    return {'story_id': story_id, 'title': title, 'part': 1, 'text': result['text'], 'model': result['model'], 'latency_ms': result.get('latency_ms')}

@app.post('/story/next/{story_id}')
def next_part(story_id: str, preferred_model: str = None, max_tokens: int = None, temperature: float = None):
    story = STORIES.get(story_id)
    if not story:
        raise HTTPException(status_code=404, detail='Story not found')
    current_parts = len(story['parts'])
    if current_parts >= story['total_parts']:
        return {'done': True, 'message': 'Story already complete', 'total_parts': story['total_parts']}
    summary_prompt = ("Provide a concise bullet summary (5 bullets) of the existing story parts to maintain coherence.\n" + "\n\n".join(story['parts']))
    summary_result = call_model(summary_prompt, prefer=preferred_model, max_tokens=256, temperature=temperature)
    summary_text = summary_result.get('text', '')[:1500]
    generation_prompt = (f"You are writing Part {current_parts+1} of a {story['total_parts']}-part fiction titled '{story['title']}'. "
                         "Maintain style, characters, and continuity. Incorporate threads from this summary and end with anticipation for next part without cliffhanger spam.\nSummary:\n" + summary_text + "\n\nPrevious Parts:\n" + "\n\n".join(story['parts']) + f"\n\nNow write Part {current_parts+1}:\n")
    part_result = call_model(generation_prompt, prefer=preferred_model, max_tokens=max_tokens, temperature=temperature)
    if 'error' in part_result:
        raise HTTPException(status_code=502, detail=part_result)
    story['parts'].append(part_result['text'].strip())
    return {'story_id': story_id, 'part': current_parts+1, 'text': part_result['text'], 'model': part_result['model'], 'latency_ms': part_result.get('latency_ms')}

@app.get('/story/{story_id}')
def get_story(story_id: str):
    story = STORIES.get(story_id)
    if not story:
        raise HTTPException(status_code=404, detail='Story not found')
    return {'story_id': story_id, **story}

@app.get('/stories')
def list_stories():
    return [{'story_id': sid, 'title': s['title'], 'parts_done': len(s['parts']), 'total_parts': s['total_parts']} for sid, s in STORIES.items()]

@app.delete('/story/{story_id}')
def delete_story(story_id: str):
    if story_id in STORIES:
        del STORIES[story_id]
        return {'deleted': story_id}
    raise HTTPException(status_code=404, detail='Story not found')

@app.get('/generate/mistral')
def gen_mistral(prompt: str, max_tokens: int = None, temperature: float = None):
    endpoint = os.getenv('MODEL_MISTRAL_ENDPOINT')
    if not endpoint:
        return {'error': 'MISTRAL endpoint not configured'}
    if max_tokens is None: max_tokens = GEN_MAX_TOKENS_DEFAULT
    if temperature is None: temperature = GEN_TEMP_DEFAULT
    payload = {
        'model': os.getenv('MISTRAL_MODEL', 'mistral-7b-instruct-v0.2'),
        'prompt': prompt,
        'max_tokens': max_tokens,
        'temperature': temperature
    }
    t0 = time.time(); r = requests.post(f"{endpoint}/v1/completions", json=payload); latency_ms = (time.time()-t0)*1000
    if not r.ok:
        return {'status_code': r.status_code, 'text': r.text[:500]}
    data = r.json(); text = ''.join(c.get('text','') for c in data.get('choices', []))
    return {'model': 'mistral', 'text': text.strip(), 'raw': data, 'latency_ms': round(latency_ms,2)}

@app.get('/generate/phi3')
def gen_phi3(prompt: str, max_tokens: int = None, temperature: float = None):
    endpoint = os.getenv('MODEL_PHI3_ENDPOINT')
    if not endpoint:
        return {'error': 'PHI3 endpoint not configured'}
    if max_tokens is None: max_tokens = GEN_MAX_TOKENS_DEFAULT
    if temperature is None: temperature = GEN_TEMP_DEFAULT
    payload = {
        'model': os.getenv('PHI3_MODEL', 'phi3:mini'),
        'prompt': prompt,
        'options': {'temperature': temperature, 'num_predict': max_tokens}
    }
    t0 = time.time(); r = requests.post(f"{endpoint}/api/generate", json=payload); latency_ms = (time.time()-t0)*1000
    if not r.ok:
        return {'status_code': r.status_code, 'text': r.text[:500]}
    data = r.json(); text = data.get('response') or data.get('generated_text') or data.get('text') or ''
    return {'model': 'phi3', 'text': text.strip(), 'raw': data, 'latency_ms': round(latency_ms,2)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)
