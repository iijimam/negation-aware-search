from sqlalchemy import create_engine,text
from typing import Any, Dict, List, Tuple,Optional
from sqlalchemy import create_engine,text
from openai import OpenAI
client = OpenAI()  # 環境変数 OPENAI_API_KEY を利用

engine = None
conn = None

# IRIS接続文字列
USER = "SuperUser"
PWD = "SYS"
HOST = "localhost"
PORT = 1972
NAMESPACE = "USER"
DATABASE_URL = f"iris://{USER}:{PWD}@{HOST}:{PORT}/{NAMESPACE}"

def initial():
    global engine, conn
    # DB接続
    engine = create_engine(DATABASE_URL,echo=False)
    if engine is None:
        engine =create_engine(DATABASE_URL,echo=True, future=True)
    if conn is None:
        conn = engine.connect()
        


def text_embedding(text :str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding
    return embedding


def search_topk(query_vec: str, topn: int, where_extra: str = "") -> List[Dict[str, Any]]:
    sql = f"""
SELECT TOP :topN
  c.DocId, c.SectionText,
  VECTOR_COSINE(c.Embedding, TO_VECTOR(:query_vec, FLOAT, 1536)) AS score_text,
  d.PatientId,d.DischargeDate
FROM Demo.DischargeSummaryChunk c, Demo.DischargeSummaryDoc d
WHERE d.DocId=c.DocId AND (c.SectionType = 'hospital_course')
  {("AND " + where_extra) if where_extra else ""}
ORDER BY score_text DESC
"""

    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"topN": topn, "query_vec": query_vec}).mappings().all()
    return rows


if __name__ == "__main__":
    initial()
    query_text="咳嗽と痰、発熱で入院しましたが、胸部画像で肺炎所見は明確ではありませんでした。対症療法で改善し、酸素投与やICU管理は不要でした"
    query_emb = text_embedding(query_text)
    query_vec = ",".join(map(str, query_emb))
    rows=search_topk(query_vec, 10)
    print(f"***その1：質問文に「酸素投与やICU管理は不要でした」 / 結果の中に「酸素を使用した」「酸素投与を短期間実施」が含まれる。\n質問文全体 >>>  {query_text} \n")
    for reco in rows:
        print(f"{str(reco["DocId"])} - {reco["SectionText"]}")
    
    query_text="高熱と咳で入院し SpO2低下のため短期間酸素投与しました。迅速検査でインフルエンザ陽性で、抗菌薬ではなく対症療法中心で改善しました。"
    query_emb = text_embedding(query_text)
    query_vec = ",".join(map(str, query_emb))
    rows=search_topk(query_vec, 10)
    print(f"----\n\n***その2：質問文に「酸素投与しました」 / 結果に「酸素投与は行わなかった」が含まれる。\n質問文 >>> {query_text} \n")
    for reco in rows:
        print(f"{str(reco["DocId"])} - {reco["SectionText"]}")