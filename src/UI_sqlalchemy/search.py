from sqlalchemy import create_engine,text
from openai import OpenAI
import json
import re
from typing import Any, Dict, List, Tuple,Optional
from string import Template
from sentence_transformers import CrossEncoder

engine = None
conn = None

# IRIS接続文字列
USER = "SuperUser"
PWD = "SYS"
HOST = "localhost"
PORT = 1972
NAMESPACE = "USER"
DATABASE_URL = f"iris://{USER}:{PWD}@{HOST}:{PORT}/{NAMESPACE}"

client = OpenAI()  # 環境変数 OPENAI_API_KEY を利用
MODEL = "gpt-4.1-mini"  # judgeは安定性重視。必要なら gpt-4.1 に上げる
TOPK = 100

#-----------------------------------
# リランキング
#-----------------------------------
Candidate = Dict[str, Any]
class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        model_name 例:
          - "BAAI/bge-reranker-v2-m3"
          - "jinaai/jina-reranker-v2-base-multilingual"
          - "Qwen/Qwen3-Reranker-0.6B" など（名称は環境に合わせて）
        """
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        text_key: str = "SectionText",
        batch_size: int = 16,
        top_n: int = 80,
    ) -> List[Candidate]:
        # 1) rerank対象を絞る（候補が多いと遅くなるので）
        cands = sorted(candidates, key=lambda x: float(x.get("vec_score", 0.0)), reverse=True)
        pool = cands[: min(top_n, len(cands))]

        # 2) (query, doc) のペアを作る
        pairs: List[Tuple[str, str]] = [(query, str(c.get(text_key, ""))) for c in pool]

        # 3) Cross-Encoderでスコア推論
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        # 4) スコアを付与して並べ替え
        for c, s in zip(pool, scores):
            c["ce_score"] = float(s)

        pool.sort(key=lambda x: float(x["ce_score"]), reverse=True)
        return pool

def initial():
    global engine, conn, reranker
    # DB接続
    engine = create_engine(DATABASE_URL,echo=False)
    if engine is None:
        engine =create_engine(DATABASE_URL,echo=True, future=True)
    if conn is None:
        conn = engine.connect()

    #リランクのインスタンス作成
    #Dockerfile内で　BAAI/bge-reranker-v2-m3　を　/opt/src/models/bge-reranker　にロード済
    reranker = CrossEncoderReranker(model_name="/opt/src/models/bge-reranker", device="cpu")        

initial()

# ===== LLM抽出プロンプト（そのまま） =====
SYSTEM_PROMPT = """あなたは医療文書から「入力テキストに明示された記載」だけに基づいて、指定フラグを構造化抽出する情報抽出器です。
出力は **JSONオブジェクトのみ** とし、未確定は **null** を使ってください。
# 最重要ルール（推論ゼロ）
- 推論・補完・常識判断は禁止。入力に「明示」されていない事項は value=null。
- 他のフラグや臨床常識からの推定は禁止。
- value を 1 または 0 にする場合、必ず evidence に根拠の原文抜粋（1〜2文以内）をそのまま入れる。
- evidence は入力からの引用で、改変しない（句読点の省略は可）。作り話は禁止。
- 「記載なし（null）」と「否定（0）」は別物。否定表現が明示されている場合のみ 0 を付ける。
- confidence は 0〜1。根拠が直接・明確なら高く、曖昧なら低くする。
- scope を必ず出力する:
  - inpatient / history / discharge_plan / unknown
# 否定の扱い（必ず考慮）
- 否定例：なし / ない / なく / 行わず / 不要 / 中止 / 見送り / せず / しない / 実施せず / 導入せず / 投与せず / 使用せず
# 判定基準（辞書）
- HasOxygenTherapy:
  - 1: 酸素投与 / 鼻カニュラ / マスク / リザーバ / 酸素◯L / 酸素投与開始 / O2投与
  - 0: 酸素なし / 酸素投与せず / 酸素不要 / 室内気で経過
- HasHFNC:
  - 1: HFNC / ハイフロー / High flow / ネーザルハイフロー
  - 0: HFNCなし / HFNC導入せず / HFNC不要
- HasNPPV:
  - 1: NPPV / CPAP / BiPAP / 非侵襲的陽圧換気
  - 0: NPPVなし / NPPV導入せず / NPPV不要
- HasIntubation:
  - 1: 挿管 / 気管挿管 / 挿管管理
  - 0: 挿管せず / 非挿管
- HasMechanicalVentilation:
  - 1: 人工呼吸器管理 / 機械換気 / ventilator / MV管理
  - 0: 人工呼吸器なし / 機械換気なし
- HasTracheostomy:
  - 1: 気管切開
  - 0: 気管切開なし
- HasICUCare:
  - 1: ICU入室 / ICU管理 / 集中治療室で管理
  - 0: ICU入室せず / ICU管理なし
- HasSepsis:
  - 1: 敗血症 / Sepsis
  - 0: 敗血症なし / 敗血症ではない（明示がある場合のみ）
- HasShock:
  - 1: ショック / 循環不全 / ショックバイタル
  - 0: ショックなし（明示がある場合のみ）
- HasVasopressor:
  - 1: 昇圧剤 / ノルアドレナリン / バソプレシン / ドパミン
  - 0: 昇圧剤なし / 使用せず（明示がある場合のみ）
- HasAKI:
  - 1: AKI / 急性腎障害 / 急性腎不全
  - 0: AKIなし（明示がある場合のみ）
- HasDialysis:
  - 1: 透析 / CHDF / HD / HDF / CRRT
  - 0: 透析なし / 導入せず（明示がある場合のみ）
- HasDiabetes:
  - 1: 糖尿病 / DM / 1型 / 2型
  - 0: 糖尿病なし（明示がある場合のみ）
- HasInsulinUse:
  - 1: インスリン導入 / 開始 / 投与
  - 0: インスリンなし / 導入せず / 中止（明示がある場合のみ）
- HasAntibioticsIV:
  - 1: 点滴抗菌薬 / 静注抗菌薬 / 抗菌薬点滴 / 静注開始
  - 0: 抗菌薬投与せず（明示がある場合のみ）
- HasAntibioticsPO:
  - 1: 内服抗菌薬 / 経口抗菌薬 / 内服へ切替
  - 0: 内服抗菌薬なし（明示がある場合のみ）
- HasSteroidSystemic:
  - 1: ステロイド投与 / PSL / プレドニゾロン / メチルプレドニゾロン（全身）
  - 0: ステロイドなし / 使用せず / 中止（明示がある場合のみ）
# 出力制約
- JSON以外を出力したら失敗です。必ず { から始まり } で終わる JSONオブジェクトのみを返してください。
- 指定された全フラグを必ず出力する（value/polarity/evidence/confidence/note を含める）。
- value が null のときは polarity="unknown", evidence=null, confidence=0.0 を基本とする
"""

OUTPUT_FORMAT2 = """
{
  "schema_version": "flags.v2",
  "doc_type": "query",
  "flags": {
    "<FlagName>": {
      "value": 1 or 0 or null,
      "polarity": "affirmed" or "negated" or "unknown",
      "evidence": "<根拠抜粋。valueがNoneならnullで可>",
      "confidence": 0.0-1.0,
      "scope": "inpatient" or "history" or "discharge_plan" or "unknown",
      "note": "<迷いがあれば短く。なければ空文字>"
    }
  }
}
"""

FLAGNAME = """
# 抽出対象フラグ（このキー名のまま全て出力）
- HasOxygenTherapy
- HasHFNC
- HasNPPV
- HasIntubation
- HasMechanicalVentilation
- HasTracheostomy
- HasICUCare
- HasSepsis
- HasShock
- HasVasopressor
- HasAKI
- HasDialysis
- HasDiabetes
- HasInsulinUse
- HasAntibioticsIV
- HasAntibioticsPO
- HasSteroidSystemic
"""



#-----------------------------------
# LLMを利用して重要用語とその状態を抽出
#-----------------------------------
def extract_flags(input_text: str, which: str) -> Dict[str, Any]:
    if which == "query":
        system_prompt = f"{SYSTEM_PROMPT}\n{OUTPUT_FORMAT2}\n{FLAGNAME}\n"
    else:
        raise ValueError("extract_flags(which) は 'query' のみ想定（ランキング用）")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text},
        ],
        max_tokens=1500,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content.strip())

#-----------------------------------
# DB FlagsJson が
#  - {"flags": {...}} wrapper
#  - {...} 直下
#  - JSON文字列
# のどれでも flags辞書（HasXXX -> {value...}）を返す
#-----------------------------------
def normalize_flags_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return {}
    if not isinstance(obj, dict):
        return {}
    if "flags" in obj and isinstance(obj["flags"], dict):
        return obj["flags"]
    if any(k.startswith("Has") for k in obj.keys()):
        return obj
    return {}

#-----------------------------------
# LLMで抽出したフラグの情報全体（Dict）を返す
#-----------------------------------
def _get_flag_obj(flags: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = flags.get(key, {})
    return v if isinstance(v, dict) else {}

#-----------------------------------
# LLMで抽出したフラグのvalue（1/0/null）を返す
#-----------------------------------
def _get_value(flags: Dict[str, Any], key: str):
    return _get_flag_obj(flags, key).get("value", None)

#-----------------------------------
# LLMで抽出したフラグのconfidenceの値を返す
#-----------------------------------
def _get_conf(flags: Dict[str, Any], key: str) -> float:
    c = _get_flag_obj(flags, key).get("confidence", 0.0)
    try:
        return float(c)
    except Exception:
        return 0.0

#-----------------------------------
# LLMで抽出したフラグのevidenceを返す（なければ空文字）
#-----------------------------------
def _get_evidence(flags: Dict[str, Any], key: str) -> str:
    ev = _get_flag_obj(flags, key).get("evidence", "")
    if ev is None:
        return ""
    return str(ev)

#-----------------------------------
# value を null に落とす（Queryガード用）
#-----------------------------------
def _set_null(flags: Dict[str, Any], key: str, note: str = "") -> None:
    obj = _get_flag_obj(flags, key)
    obj["value"] = None
    obj["polarity"] = "unknown"
    obj["evidence"] = None
    obj["confidence"] = 0.0
    # scope は残す（抽出結果の情報としては有用）
    if note:
        obj["note"] = note
    flags[key] = obj


# LLMから抽出した重要用語の状態からWHERE作成時に使用する 
# 強フラグ　WHERE フラグ=1
STRONG_FLAGS = [
    "HasICUCare",
    "HasNPPV",
    "HasMechanicalVentilation",
    "HasIntubation",
    "HasDialysis",
    "HasVasopressor",
]
# 中間フラグ（コードでは未使用）
MID_FLAGS = ["HasHFNC"]
# 弱フラグ（コードでは未使用）
WEAK_FLAGS = [
    "HasOxygenTherapy",
    "HasSepsis",
    "HasShock",
    "HasAKI",
    "HasDiabetes",
    "HasInsulinUse",
    "HasAntibioticsIV",
    "HasAntibioticsPO",
    "HasSteroidSystemic",
]


# LLMから抽出した重要用語の状態からWHEREを作成する際の 
# 0(否定) を採用してよい「強い否定」
STRONG_NEG_PAT = re.compile(
    r"(?:"
    r"なし|無い|ない|なく|認めず|認めない|否定|ではない|"
    r"未施行|未実施|未導入|"
    r"不要|必要なし|"
    r"行わず|行わない|行わなかった|行っていません|"
    r"使わず|使わない|使わなかった|"          # ★追加
    r"実施せず|導入せず|投与せず|使用せず|"
    r"中止|見送り|非該当|"
    r"室内気で経過|酸素なし"
    r")"
)

# LLMから抽出した重要用語の状態からWHEREを作成する際の 
# 「主ではない」系（否定ではなく相対評価なので 0 で絞らない nullとする）
RELATIVE_NOT_MAIN_PAT = re.compile(
    r"(?:"
    r"主ではない|主因ではない|主体ではない|中心ではない|"
    r"優位ではない|優位でない|"
    r"第一選択ではない|メインではない|"
    r"補助的|補助的でした|"
    r"感染より.*が主因|感染より.*増悪が主因|"
    r"肺炎としての.*は主ではない"
    r")"
)


#否定(0)を採用する際の「一般ルール」を優先し、
#個別疾患（例：敗血症）専用の正規表現に依存しない形に寄せる。
#
#value=0 を採用するには原則として
#  (a) evidence に強い否定表現がある
#  (b) 対象概念の mention（同義語）が text/evidence に明示されているを満たす（ドメイン差し替えは mention 辞書だけで吸収）。
#
# フラグごとの「明示語（mention）」。他ドメイン展開時はここを差し替える。
# ※237文字程度の短文なので、ここは単純な部分一致で十分。
FLAG_MENTION_KEYWORDS: Dict[str, List[str]] = {
    "HasOxygenTherapy": ["酸素", "O2", "室内気"],
    "HasHFNC": ["HFNC", "ハイフロー", "High flow", "ネーザルハイフロー"],
    "HasNPPV": ["NPPV", "CPAP", "BiPAP", "非侵襲"],
    "HasIntubation": ["挿管", "気管挿管"],
    "HasMechanicalVentilation": ["人工呼吸", "機械換気", "ventilator", "MV"],
    "HasTracheostomy": ["気管切開"],
    "HasICUCare": ["ICU", "集中治療"],
    "HasSepsis": ["敗血症", "菌血症", "sepsis"],
    "HasShock": ["ショック", "循環不全"],
    "HasVasopressor": ["昇圧", "ノルアド", "ノルアドレナリン", "バソプレシン", "ドパミン"],
    "HasAKI": ["AKI", "急性腎"],
    "HasDialysis": ["透析", "CHDF", "CRRT", "HD", "HDF"],
    "HasDiabetes": ["糖尿病", "DM", "1型", "2型"],
    "HasInsulinUse": ["インスリン"],
    "HasAntibioticsIV": ["点滴", "静注", "抗菌"],
    "HasAntibioticsPO": ["内服", "経口", "抗菌"],
    "HasSteroidSystemic": ["ステロイド", "PSL", "プレド", "メチルプレド"],
}


#-----------------------------------
# 第1引数の文字列にFLAG_MENTION_KEYWORDSの指定フラグの値が含まれているかどうか
#-----------------------------------
def _contains_any(text: str, keywords: List[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    for kw in keywords:
        if kw and kw.lower() in t:
            return True
    return False

#-----------------------------------
# value=0 を採用してよいか（一般ルール）
# 強い否定であるかどうかLLMチェック後のconfidence値でチェック
#-----------------------------------
def _should_accept_negation(flag_name: str, flag_obj: Dict[str, Any], full_text: str) -> bool:
    if not isinstance(flag_obj, dict):
        return False

    v = flag_obj.get("value", None)
    if v != 0:
        return False

    ev = str(flag_obj.get("evidence") or "")

    # 相対評価/主ではないは「否定」ではないので落とす
    if ev and RELATIVE_NOT_MAIN_PAT.search(ev):
        return False

    # evidence に強い否定がない 0 は採用しない（保守的）
    if ev and not STRONG_NEG_PAT.search(ev):
        return False

    # mention がなければ 0 を採用しない（0の誤爆を防ぐ）
    kws = FLAG_MENTION_KEYWORDS.get(flag_name, [])
    if kws and not (_contains_any(ev, kws) or _contains_any(full_text, kws)):
        return False

    return True

#-----------------------------------
# 質問文（Query）のフラグからWHERE作成時の情報を落としすぎないようにするための安全策
# - value=1 は基本そのまま採用（=肯定は誤爆してもその他処理で救えることが多い）
# - value=0 は誤爆が致命傷になりやすいので、一般ルールで厳格にガードしてダメなら null へ
#-----------------------------------
def postprocess_query_flags(query_text: str, flags: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(flags, dict):
        return {}

    for k, obj in list(flags.items()):
        if not (isinstance(k, str) and isinstance(obj, dict)):
            continue
        if obj.get("value", None) == 0:
            if not _should_accept_negation(k, obj, query_text):
                _set_null(flags, k, note="Queryガード: 0(否定)の根拠が弱い/mention不足/相対表現のためnullへ")

    return flags


#-----------------------------------
# 質問文（Query）のフラグからWHERE作成
# フラグは OR で条件指定する
#-----------------------------------
def build_optional_filters(query_text: str, query_flags: Dict[str, Any]) -> str:
    q = normalize_flags_dict(query_flags)
    conds = []

    # 強フラグ = 1 -> WHEREに指定
    for k in STRONG_FLAGS:
        if _get_value(q, k) == 1:
            conds.append(f"c.{k} = 1")

    # HasOxygenTherapy が「明確に 0」の場合
    oxy = q.get("HasOxygenTherapy", {})
    if oxy.get("value") == 0:
        ev = (oxy.get("evidence") or "")
        conf = float(oxy.get("confidence", 0.0))
        # 強い否定 + mention + 高 confidence のみ採用
        if conf >= 0.9 and _should_accept_negation("HasOxygenTherapy", oxy, query_text):
            conds.append("c.HasOxygenTherapy IS NULL OR c.HasOxygenTherapy = 0")
    if not conds:
        return ""

    return "(" + " OR ".join(conds) + ")"


#-----------------------------------
# 2つの類似検索結果のマージ（DocIdをキーにチェック）
#-----------------------------------
def merge_unique_by_docid(*lists_: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for lst in lists_:
        for r in lst:
            docid = r.get("DocId")
            if docid is None:
                continue
            docid = str(docid)   # 型を正規化
            if docid in seen:
                continue
            seen.add(docid)
            out.append(r)
    return out

#-----------------------------------
# 2回の類似検索結果と質問文を利用して
# 特定用語に対する否定（0）が質問文に食い違いがないかのチェック
# (絶対に入れたくない矛盾を除外する：WHEREで取り除けないもの、マージ後の混入を除外)
# 質問文が強く否定(0)なのにDB側で肯定(1)している場合、結果から取り除く
# （LLMが回答したconfidenceの値を利用：パラメータでどの値以上なら取り除くを指定可）
# DB側が value==1 でも confidence が低い場合は誤爆の可能性があるため除外しない
#-----------------------------------
def hard_exclude_contradictions(
    results,
    query_text,
    query_flags,
    hard_flags=None,
    conf_th=0.9,         # query側
    doc_conf_th=0.8,     # doc側
    bypass_accept_if_query_conf_ge=0.99,
):
    q = normalize_flags_dict(query_flags)
    hard_flags = hard_flags or ["HasOxygenTherapy"]

    out = results
    for flag_name in hard_flags:
        fq = q.get(flag_name, {})
        if not (isinstance(fq, dict) and fq.get("value", None) == 0):
            continue

        q_conf = float(fq.get("confidence", 0.0) or 0.0)
        if q_conf < conf_th:
            continue

        # 超高信頼なら _should_accept_negation を必須にしない
        if q_conf < bypass_accept_if_query_conf_ge:
            if not _should_accept_negation(flag_name, fq, query_text):
                continue

        kept = []
        for r in out:
            d = normalize_flags_dict(r.get("FlagsJson"))
            dv = _get_value(d, flag_name)

            d_obj = d.get(flag_name, {})
            d_conf = 0.0
            if isinstance(d_obj, dict):
                try:
                    d_conf = float(d_obj.get("confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    d_conf = 0.0

            # 確からしい「1」だけ落とす
            if dv == 1 and d_conf >= doc_conf_th:
                continue

            kept.append(r)

        out = kept

    return out



#-----------------------------------
# Embedding
#-----------------------------------
def text_embedding(text_: str):
    resp = client.embeddings.create(model="text-embedding-3-small", input=text_)
    return resp.data[0].embedding

#-----------------------------------
# 類似検索結果の出力作成（FlagJsonの中身をDictにする）
#-----------------------------------
def make_output_topk(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        rr = dict(r)
        # rerank側で統一的に参照できるよう、ベクトル類似度を vec_score にも入れる
        if "vec_score" not in rr and "score_text" in rr:
            rr["vec_score"] = rr.get("score_text")
        if isinstance(rr.get("FlagsJson"), str):
            try:
                rr["FlagsJson"] = json.loads(rr["FlagsJson"])
            except Exception:
                rr["FlagsJson"] = None
        out.append(rr)
    return out



def search_topk(query_vec: str, topn: int, where_extra: str = "") -> List[Dict[str, Any]]:
    sql = f"""
SELECT TOP :topN
  c.DocId, c.SectionText, c.FlagsJson,
  VECTOR_COSINE(c.Embedding, TO_VECTOR(:query_vec, FLOAT, 1536)) AS score_text,
  d.PatientId,d.DischargeDate
FROM Demo.DischargeSummaryChunk c, Demo.DischargeSummaryDoc d
WHERE d.DocId=c.DocId AND (c.SectionType = 'hospital_course')
  {("AND " + where_extra) if where_extra else ""}
ORDER BY score_text DESC
"""
    print(sql)
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"topN": topn, "query_vec": query_vec}).mappings().all()
    return make_output_topk([dict(r) for r in rows])


#-----------------------------------
# フラグに対する値の数を取得
#-----------------------------------
def count_doc_flag_value(results, flag_name, value):
    n = 0
    for r in results:
        d = normalize_flags_dict(r.get("FlagsJson"))
        if _get_value(d, flag_name) == value:
            n += 1
    return n

#-----------------------------------
#類似検索関数
# 1) TOP100
# 2) LLMを利用して質問文から重要用語（フラグ）とその状態を抽出
# 3) 2)からWHEREの条件指定ができるか確認
#    3)の結果からあればTOP50の類似検索実行
#    TOP50実施後はTOP100と結果をマージ
# 4) リランク実行（CrossEncoding）
#    リランク後のTOP3を利用してLLM as a Juedge用のJSON作成
#-----------------------------------
def get_simirality_ranking(query_text):
    result = {}

    # 1) ベクトル検索 TOP100
    query_emb = text_embedding(query_text)
    query_vec = ",".join(map(str, query_emb))
    results100 = search_topk(query_vec, 100)

    # 2) LLMで query_flags 抽出
    query_flags = extract_flags(query_text, "query")["flags"]
    query_flags = postprocess_query_flags(query_text,query_flags) #抽出後の再調整

    # 3) 条件文ができるか確認（あればTOP50追加）
    conds = build_optional_filters(query_text,query_flags)

    if conds:
        results50 = search_topk(query_vec, 50, conds)
        merged = merge_unique_by_docid(results100, results50)
        filtered_n = len(results50)
        where_sql = conds
    else:
        merged = results100
        filtered_n = 0
        where_sql = ""

    #マージ直後の数
    merged_raw_n = len(merged)
    # 特定用語のvalue=1の数
    o2_pos_before = count_doc_flag_value(merged, "HasOxygenTherapy", 1)
    #（必要に応じて）特定のフラグに対して質問文とDB登録のフラグ値が強く食い違うものを取り除く
    merged = hard_exclude_contradictions(
        merged, query_text, query_flags,
        hard_flags=["HasOxygenTherapy"],
        conf_th=0.9,
        doc_conf_th=0.8,   # 例：0.7にするともっと落ちる／0.9にすると落ちにくい
        )
    merged_final_n = len(merged)
    dropped_n=merged_raw_n-merged_final_n
    # hard_exclude_contradictions()実行後の特定用語のvalue=1の数
    o2_pos_after  = count_doc_flag_value(merged, "HasOxygenTherapy", 1)
    # 4) リランク
    reranked = reranker.rerank(query_text, merged, top_n=50)
    top3 = reranked[:3]

    # 5) judge 用 top3 生成（FlagsJson は dictのまま）
    judge_top3 = []
    search_top3 = []
    for r in top3:
        doc_flags = r.get("FlagsJson")
        if isinstance(doc_flags, dict) and "flags" in doc_flags and isinstance(doc_flags["flags"], dict):
            doc_flags = doc_flags["flags"]

        judge_top3.append(
            {
                "DocId": r.get("DocId"),
                "score_text": r.get("score_text"),
                "score_rerank": r.get("ce_score"),
                #"score_text_norm": r.get("score_text_norm"),
                #"final_score": r.get("final_score"),
                # SectionText は使わない方針なら保存しない（必要ならコメント解除）
                #"SectionText": r.get("SectionText"),
                "FlagsJson": doc_flags,
            }
        )
        #画面表示用情報
        search_top3.append(
            {
                "DocId": r.get("DocId"),
                "score_text": r.get("score_text"),
                "score_text_norm": r.get("score_text_norm"),
                "final_score": r.get("final_score"),
                "PatientId": r.get("PatientId"),
                "DischargeDate": r.get("DischargeDate"),
                "SectionText": r.get("SectionText"),
                "FlagsJson": doc_flags,
            }
        )

    result.update(
        {
            "query_text": query_text,
            "query_flags": query_flags,
            "stage1": {
                "base_k": 100,
                "base_n": len(results100),
                "filtered_k": 50,
                "filtered_n": filtered_n,
                "where_sql": where_sql,
                "merged_raw_n": merged_raw_n,
                "excluded_n":dropped_n,
                "merged_final_n":merged_final_n,
                "o2_pos_before": o2_pos_before,
                "o2_pos_after": o2_pos_after,
            },
            "ranked_top3": judge_top3,
        }
    )

    return result,search_top3

if __name__ == "__main__":
    initial()