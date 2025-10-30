# fix_to_jsonl.py
import json, sys, pathlib

src = pathlib.Path("CORY_withRAG/data/research_seeds.jsonl")  # ←元ファイル（今のパス）
dst = pathlib.Path("CORY_withRAG/data/research_seeds.fixed.jsonl")  # 変換後に出す先

text = src.read_text(encoding="utf-8", errors="ignore")
# 全角の波括弧を半角に正規化
text = text.replace("｛", "{").replace("｝", "}")

# 連結JSONにも、行区切りJSONLにも対応するストリームデコーダ
dec = json.JSONDecoder()
objs = []
i = 0
n = len(text)
while i < n:
    # 空白スキップ
    while i < n and text[i].isspace():
        i += 1
    if i >= n:
        break
    try:
        obj, j = dec.raw_decode(text, i)
        objs.append(obj)
        i = j
    except json.JSONDecodeError:
        # 行ごとのJSONLも試す
        nl = text.find("\n", i)
        if nl == -1:
            # 最後の行
            line = text[i:].strip()
            if line:
                try:
                    objs.append(json.loads(line))
                except Exception:
                    pass
            break
        else:
            line = text[i:nl].strip()
            if line:
                try:
                    objs.append(json.loads(line))
                except Exception:
                    pass
            i = nl + 1

# topic=="EOF" を除外し、プロンプトが作れないものは合成
def to_prompt(o: dict) -> str | None:
    if not isinstance(o, dict):
        return None
    if o.get("topic") == "EOF":
        return None
    for k in ["prompt", "problem", "seed", "task", "instruction", "input", "title", "query"]:
        v = o.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    topic = o.get("topic", "")
    problem = o.get("problem", "")
    constraints = o.get("constraints", "")
    ctx = o.get("context", [])
    ctx_titles = []
    if isinstance(ctx, list):
        for it in ctx[:3]:
            if isinstance(it, dict):
                t = it.get("title")
                if isinstance(t, str) and t.strip():
                    ctx_titles.append(t.strip())
    parts = []
    if topic: parts.append(f"Topic: {topic}")
    if problem: parts.append(f"Problem: {problem}")
    if constraints: parts.append(f"Constraints: {constraints}")
    if ctx_titles: parts.append("Context: " + "; ".join(ctx_titles))
    parts.append("Generate 3 concrete research ideas with clear evaluation plans.")
    return "\n".join(parts)

prompts = []
for o in objs:
    p = to_prompt(o)
    if p:
        prompts.append({"prompt": p})

if not prompts:
    prompts = [{"prompt": "Give me three creative research ideas that connect two distant fields (LLMs × RL)."}]

with dst.open("w", encoding="utf-8") as f:
    for row in prompts:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[fix_to_jsonl] objects_in:{len(objs)} prompts_out:{len(prompts)} -> {dst}")
