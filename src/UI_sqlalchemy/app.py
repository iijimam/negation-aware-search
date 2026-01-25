# /home/irisowner/.local/bin/streamlit run /src/UI_sqlalchemy/app.py --server.port 8090 --logger.level=debug
#
import streamlit as st
from openai import OpenAI
from typing import Any, Dict, List, Tuple,Optional
import json
import time,datetime
import sys
sys.path+=["/src/UI_sqlalchemy"]
import search

client = OpenAI()  # 環境変数 OPENAI_API_KEY を利用
MODEL = "gpt-4o-mini"  #すでに決まった Top3 を、ルールに沿って機械的に判定するため、4.1-miniではないモデルを利用

SYSTEM_PROMPT = """
あなたはランキング候補(top3)の妥当性を判定する Judge です。
判定に使う情報は「query_flags と ranked_top3[].FlagsJson の value」のみ。
query_text / SectionText / evidence の文章を読んで再解釈してはいけません。

# 値の扱い
value は 1/0/null。null は「不明」であり矛盾ではない。

# mismatch は強フラグ矛盾だけ（これ以外で mismatch 禁止）
強フラグ = HasICUCare, HasNPPV, HasMechanicalVentilation, HasIntubation, HasDialysis, HasVasopressor
mismatch 条件は次の2つのみ:
(A) query=1 かつ doc=0
(B) query=1 かつ doc!=1（docがnull/0を含む）
※強フラグ以外は(A)(B)を適用しない。

# 弱フラグは decision に使わない（絶対）
弱フラグ = HasOxygenTherapy, HasAntibioticsIV, HasAntibioticsPO, HasSteroidSystemic
弱フラグは「説明に書いてよい」だけで、is_similar_enough の判定根拠にしてはいけない。

# verdict ルール
- match: 強フラグ矛盾がなく、主要フラグの整合が高い
- partial: 強フラグ矛盾はないが、情報が薄く確信が弱い
- mismatch: 強フラグ矛盾(A)(B)が1つでもあれば必ず mismatch

# decision の作り方（強制・例外なし）
- decision.top_doc_id = ranking[0].doc_id
- decision.is_similar_enough = (ranking[0].verdict != "mismatch")
- decision.summary は ranking[0] の verdict と、強フラグ矛盾の有無だけを短く述べる。
  「確認されている」「〜が行われた」等の臨床イベント断定は禁止（value=1 のフラグ名だけ書く）。

# reasons の status 判定（機械的ルール）
各フラグごとに以下で status を決める（例外なし）:

(1) query が null または doc が null → status=neutral
(2) query が 0/1 で doc が 0/1 かつ一致 → status=match
(3) query が 0/1 で doc が 0/1 かつ不一致 → status=contradict
ただし (3) の "contradict" を許可するのは「強フラグ」かつ query=1 & doc!=1 のときのみ。
それ以外の不一致はすべて neutral とする（弱フラグは決して contradict にしない）。

# 出力
必ずJSONのみ:
{
  "decision": {"top_doc_id": <int>, "is_similar_enough": <bool>, "confidence": <0-1>, "summary": "<短い日本語>", "missing_info": []},
  "ranking": [{"doc_id": <int>, "rank": 1, "relevance": <0-1>, "verdict": "match|partial|mismatch", "reasons": [...]}, ...]
}
"""

JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "ranking"],
    "properties": {
        "decision": {
            "type": "object",
            "additionalProperties": False,
            "required": ["top_doc_id","is_similar_enough", "confidence", "summary", "missing_info"],
            "properties": {
                "top_doc_id": {"type": "number"},
                "is_similar_enough": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "summary": {"type": "string"},
                "missing_info": {"type": "array", "items": {"type": "string"}},
            },
        },
        "ranking": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["doc_id", "rank", "relevance", "verdict", "reasons"],
                "properties": {
                    "doc_id": {"type": "number"},
                    "rank": {"type": "number"},
                    "relevance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "verdict": {"type": "string", "enum": ["match", "partial", "mismatch"]},
                    "reasons": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                },
            },
        },
    },
}

def build_user_prompt(item: dict) -> str:
    return f"""次の入力(JSON)について判定してください。

入力(JSON):
<<<
{json.dumps(item, ensure_ascii=False)}
>>>

最重要: decision は「rank1（最上位候補）に対する結論」とする。
- decision.is_similar_enough は rank1 の候補がクエリと類似しているかのみで判定する。
- rank2/3 に矛盾があっても decision を False にしてはいけない（decision は rank1 のみを見る）。
- decision.summary / decision.reasons は rank1 の内容だけに基づいて書く。rank2/3 に言及しない。
- decision.top_doc_id と decision.top_rank を必ず出力する（top_rankは常に1）。

decision と verdict の整合:
- rank1.verdict が match のとき、decision.is_similar_enough は必ず true
- rank1.verdict が mismatch のとき、decision.is_similar_enough は必ず false
- rank1.verdict が partial のときは true/false どちらも可（confidenceで調整）

判定の優先順位:
(1) クエリで明示されたフラグ(0/1)の一致・矛盾（最重要）
(2) ICU/NPPV/MVなど重症度イベントの矛盾
(3) 疾患（肺炎など）の一致
(4) 転帰や細部は補助
nullは「不明」であり、0とは違う。

制約:
- rankingは必ず3件(入力の3候補を必ず含める)
- reasonsは各候補 最大5個、短く具体的に、クエリで明示されたフラグ(0/1)と疾患・重症度を中心に
- relevanceは0.0〜1.0
- relevanceは相対値。rank1>rank2>rank3 となるように差をつけること（同点禁止）
- verdictの定義は以下の通り:
    - match: 強フラグ(ICU/NPPV/MV/挿管/透析/昇圧剤)の矛盾がなく、主要情報が整合
    - partial: 強フラグ矛盾はないが、情報が薄い/疾患が不明/弱フラグが揃わず確信が弱い
    - mismatch: 強フラグに限り、クエリで明示(query=1)された強フラグが doc で満たされない場合のみ
      (query=1 & doc!=1)
※HasOxygenTherapy を含む弱フラグでは mismatch にしない
- クエリで未言及(value=null)の弱フラグ(Sepsis, Shock, AKI, Diabetes, InsulinUse, AntibioticsIV/PO, SteroidSystemic)は、verdict を下げる主要因にしない。
- クエリで未言及のフラグ（酸素など）で mismatch にしない
- decision.top_doc_id は ranking[0].doc_id(rank=1の候補)と同じ値にする。
- JSON以外は禁止
"""

def call_judge(
    client: OpenAI,
    model: str,
    item: dict,
    temperature: float = 0.0,
    max_output_tokens: int = 900,
    retries: int = 4,
) -> dict:
    last_err = None
    user_prompt = build_user_prompt(item)
    print(item)

    for attempt in range(1, retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                #  JSON Schemaで出力を制約
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "judge_result",
                        "strict": True,
                        "schema": JUDGE_SCHEMA,
                    }
                },
            )

            # 返ってきたテキスト(JSON)をパース
            return json.loads(resp.output_text)

        except Exception as e:
            last_err = e
            time.sleep(min(6.0, 0.6 * (2 ** (attempt - 1))))

    raise RuntimeError(f"Judge failed: {last_err}") from last_err

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {i}: {e}") from e
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_flagsjson_to_dict(flagsjson: Any) -> Dict[str, Any]:
    """
    入力 flagsjson が
      - flags.v2 形式の dict
      - 旧形式 list[{"FlagName":..., "Value":...}]
      - JSON文字列（dict or list）
    のどれでも受けて、最終的に {"HasX": {"value": ...}, ...} を返す
    """
    if flagsjson is None:
        return {}

    # 1) JSON文字列ならパースを試す
    if isinstance(flagsjson, str):
        s = flagsjson.strip()
        if not s:
            return {}
        try:
            flagsjson = json.loads(s)
        except json.JSONDecodeError:
            # JSON文字列でないなら諦めて空
            return {}

    # 2) すでに dict の場合
    if isinstance(flagsjson, dict):
        # flags.v2 の入れ子（{"flags": {...}}）なら中身を返す
        if "flags" in flagsjson and isinstance(flagsjson["flags"], dict):
            return flagsjson["flags"]
        # すでに {"HasX": {"value": ...}} 形式ならそのまま
        return flagsjson

    # 3) list の場合（旧形式想定）
    if isinstance(flagsjson, list):
        d: Dict[str, Any] = {}
        for it in flagsjson:
            if not isinstance(it, dict):
                continue
            k = it.get("FlagName")
            if not k:
                continue
            d[k] = {"value": it.get("Value")}
        return d

    # 4) その他は空
    return {}

def excerpt(s: str, max_chars: int = 700) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "…(truncated)"

from datetime import date

def fmt_date(d):
    if d is None:
        return None
    if isinstance(d, date):
        return d.isoformat()  # 'YYYY-MM-DD'
    return str(d)

st.set_page_config(page_title="退院時サマリ類似検索", layout="wide")
st.title("退院時サマリ類似検索＋リランク＋LLM as judge")


# 入力欄
if query := st.chat_input("類似検索用の質問を入力してください>>"):
    st.markdown(f"### 入力質問:\n{query}")
    with st.spinner("LLMでフラグ抽出＋類似検索、リランク実行中...しばらくお待ちください。"):
        ranking_result,search_top3 = search.get_simirality_ranking(query)
        with st.expander("🔍 ベクトル検索結果（リランク後）：デバッグ", expanded=False):
            search_map = {
                r["DocId"]: r
                for r in search_top3
            }
            ranked_top3_with_text = []
            for r in ranking_result["ranked_top3"]:
                docid = r["DocId"]
                src = search_map.get(docid, {})

                ranked_top3_with_text.append({
                    "PatientId": src.get("PatientId"),
                    "DischargeDate": fmt_date(src.get("DischargeDate")),
                    "SectionText": src.get("SectionText"),
                    **r,  # judge_top3 の中身（ranking_resultの中身）
                })
            ranking_out={
                "query_text": ranking_result["query_text"],
                "query_flags": ranking_result["query_flags"],
                "stage1": ranking_result["stage1"],
                "ranking":ranked_top3_with_text
            }
            st.write(ranking_out)

    # LLM as a Judge
    with st.spinner("審査中...しばらくお待ちください。"):
        #ランキングの3件
        top3_raw = ranking_result.get("ranked_top3")
        if not isinstance(top3_raw, list) or len(top3_raw) != 3:
            raise ValueError(f"ranked_top3 must have exactly 3 items.")
        out_rows: List[Dict[str, Any]] = []
        # ===== ここで judge 用に FlagsJson を dict 化する =====
        top3_for_judge = []
        for c in top3_raw:
            c2 = dict(c)
            if isinstance(c.get("FlagsJson"), str):
                print(f"FlagsJson is str (DocId={c.get('DocId')}), head={c.get('FlagsJson')[:80]}")
            c2["FlagsJson"] = normalize_flagsjson_to_dict(c.get("FlagsJson"))
            top3_for_judge.append(c2)

        item_for_judge = {
            "query_flags": ranking_result["query_flags"],
            "ranked_top3": [
                {
                    "DocId": c["DocId"],
                    "FlagsJson": c["FlagsJson"],
                }
                for c in top3_for_judge
            ]
        }
        # =====================================================

        temperature = 0.0  # 固定
        judge_result = call_judge(
            client=client,
            model=MODEL,
            item=item_for_judge,
            temperature=temperature,
            max_output_tokens=900,
        )

        out_rows.append({
            "query_text": ranking_result.get("query_text",""),
            "ranked_top3_docids": [c.get("DocId") for c in top3_raw],
            "stage1": ranking_result.get("stage1", {}),
            "ranked_top3_meta": [
                {
                    "DocId": c.get("DocId"),
                    "score_text": c.get("score_text"),
                    "score_text_norm": c.get("score_text_norm"),
                    "final_score": c.get("final_score"),
                } for c in top3_raw
            ],
            "judge_result": judge_result,
            "meta": {"model": MODEL, "temperature": temperature},
        })

        # 審査結果
        st.markdown("### 🏆 審査結果（LLMによるランキングの正しさ判定）")
        # 表示内容：1位のDocIdから主要情報を抜粋
        ranking = judge_result.get("ranking", [])
        if len(ranking) == 0:
            st.write("審査結果がありません。")

        else:
            is_similar = judge_result.get("decision", {}).get("is_similar_enough")
            if is_similar is True:
                st.success("✅ ランキングは正しい")
            elif is_similar is False:
                st.error("❌ ランキングは正しくない")
            else:
                st.info("ℹ️ 判定できず")
            
            st.markdown(f"**{judge_result.get('decision').get('summary')}**")
            top_rank = ranking[0]
            top_docid = top_rank.get("doc_id")
            top_candidate = next((c for c in search_top3 if c["DocId"] == top_docid), None)
            if top_candidate:
                st.markdown(f"#### ❓質問文：{query}")
                st.markdown(f"**☆最も類似している候補☆ DocId: {top_docid}／{excerpt(top_candidate.get('SectionText'))}**")
                ranktbl = []
                ranktbl.append("ランキング|DocId | 患者ID | 退院日 | セクション内容抜粋")
                ranktbl.append("--| -- | -- | -- | --")
                num=0
                for reco in search_top3:
                    num+=1
                    ranktbl.append(
                        f"{num}|{reco.get('DocId')}|{reco.get('PatientId')}|{reco.get('DischargeDate')}|{reco.get('SectionText')}"                        
                    )
                st.markdown("\n".join(ranktbl))
            else:
                st.write("最も類似している候補の詳細が見つかりません。")

        with st.expander("🔍 ランキング審査結果：デバッグ", expanded=False):
            st.write(judge_result)
