#!/usr/bin/env python3
"""Re-generate specific failed pair indices for a persona.

Usage: python3 fix_pairs.py --persona marigold --indices 4,5,8,10,11,14,15
       python3 fix_pairs.py --persona bramble --indices 16
"""
import argparse
import json
import os
import sys
import urllib.request

LMSTUDIO_URL = "http://192.168.137.1:1234/v1/chat/completions"
MODEL_ID = "gemma-3-12b-it"

# Read system prompt from YAML card
def get_system_prompt(persona):
    import yaml
    path = f"/home/node/github/shanevcantwell/semantic-forge/docs/experiments/persona-dpo-probe/cards/{persona}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)["system_prompt"]

PAIR_PROMPT_TEMPLATE = """You are {persona_name}. Your system instructions are provided above. A user has asked the following question — respond as yourself, giving your natural answer (the chosen response). Then, also provide an alternative response that a generic "helpful assistant" would give to the SAME question (the rejected response) — this should be more verbose, warmer, hedged with qualifiers like "maybe," "perhaps," "I think," and include encouraging language.

User question: {scenario}

Output format (JSON only):
{{
  "chosen": "<your natural persona response as {persona_name}>",
  "rejected": "<generic helpful assistant alternative, warmer/more hedged>"
}}"""


def call_gemma(messages, temperature=0.8, max_tokens=512):
    payload = {"model": MODEL_ID, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    req = urllib.request.Request(LMSTUDIO_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"].strip()


def extract_json_robust(content):
    """More robust JSON extraction that handles trailing text after the closing brace."""
    import re
    
    # Strip markdown code fences
    content = content.strip()
    if "```json" in content:
        start = content.index("```json") + 7
        end = content.rfind("```")
        content = content[start:end].strip()
    elif "```" in content:
        start = content.index("```") + 3
        end = content.rindex("```")
        content = content[start:end].strip()
    
    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Find the JSON object and extract just it (handle trailing text)
    brace_start = content.find("{")
    if brace_start == -1:
        raise ValueError(f"No JSON found:\n{content[:300]}")
    
    depth = 0
    in_string = False
    escape_next = False
    
    for i, c in enumerate(content[brace_start:], start=brace_start):
        if escape_next:
            escape_next = False
            continue
        if c == '\\':
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                json_str = content[brace_start:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    # Try cleaning up common issues in the extracted JSON
                    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', json_str)  # Remove control chars
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError as e2:
                        raise ValueError(f"JSON parse failed even after cleanup:\n{json_str[:500]}\nerror: {e2}")
    
    raise ValueError(f"Unbalanced JSON in response:\n{content[:300]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", required=True)
    parser.add_argument("--indices", required=True, help="comma-separated list of indices to re-generate")
    args = parser.parse_args()
    
    persona = args.persona
    indices = [int(i) for i in args.indices.split(",")]
    
    # Load stimuli
    stimuli_path = f"/home/node/github/shanevcantwell/semantic-forge/docs/experiments/persona-dpo-scale/run0/scenarios/bramble/stimuli.txt"
    with open(stimuli_path) as f:
        scenarios = [line.strip() for line in f if line.strip()]
    
    system_prompt = get_system_prompt(persona)
    
    # Verify gemma is up
    try:
        urllib.request.urlopen("http://192.168.137.1:1234/v1/models", timeout=5).read()
    except Exception as e:
        print(f"ERROR: LM Studio unreachable — {e}", file=sys.stderr)
        sys.exit(2)
    
    outdir = f"/home/node/github/shanevcantwell/semantic-forge/docs/experiments/persona-dpo-scale/run0/scenarios/{persona}/pairs"
    
    for idx in indices:
        if idx >= len(scenarios):
            print(f"[fix] {persona} idx={idx}: scenario out of range ({len(scenarios)} stimuli)", file=sys.stderr)
            continue
        
        scenario = scenarios[idx]
        pair_prompt = PAIR_PROMPT_TEMPLATE.format(persona_name=persona.capitalize(), scenario=scenario)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair_prompt},
        ]
        
        raw_response = call_gemma(messages, temperature=0.85, max_tokens=640)  # slightly higher temp + more tokens for marigold's verbose style
        
        try:
            data = extract_json_robust(raw_response)
            chosen = data.get("chosen", "").strip()
            rejected = data.get("rejected", "").strip()
            
            if not chosen or not rejected:
                print(f"[fix] {persona} idx={idx}: EMPTY after parse (c={len(chosen)} r={len(rejected)})", file=sys.stderr)
                continue
            
            out_row = {
                "prompt": scenario,
                "chosen": chosen,
                "rejected": rejected,
                "_persona": persona,
                "_scenario_index": idx,
                "_embedding_distance_chosen_rejected": -1.0,  # will recompute later when :8082 is stable
                "_isError": False,
            }
            
            outpath = f"{outdir}/{idx}.json"
            with open(outpath, "w") as f:
                json.dump(out_row, f)
            
            print(f"[fix] {persona} idx={idx}: ✓ chosen={len(chosen)}c rejected={len(rejected)}c -> {outpath}", file=sys.stderr)
            
        except ValueError as e:
            print(f"[fix] {persona} idx={idx}: PARSE_ERROR — {e}\nraw: {raw_response[:300]}", file=sys.stderr)


if __name__ == "__main__":
    main()
