import os
import json
import argparse
from tqdm import tqdm
import pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer

# 初始化向量模型和 Qdrant 客戶端
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
qdrant = QdrantClient("http://localhost:6333")
embedding_dim = 384  # 根據選定模型的輸出維度進行設置

# 創建向量資料庫
qdrant.recreate_collection(
    collection_name="documents_collection",
    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
)

# 將 PDF 讀取並向量化
def load_and_store_data(source_path, category):
    masked_file_ls = os.listdir(source_path)
    corpus_dict = {}
    for file in tqdm(masked_file_ls):
        file_id = int(file.replace('.pdf', ''))
        file_path = os.path.join(source_path, file)
        text = read_pdf(file_path)
        corpus_dict[file_id] = text

        # 向量化並存入 Qdrant
        vector = model.encode(text).tolist()
        qdrant.upsert(
            collection_name="documents_collection",
            points=[
                PointStruct(
                    id=file_id,
                    vector=vector,
                    payload={"category": category, "file_id": file_id}
                )
            ]
        )
    return corpus_dict

# 讀取單個 PDF 文件並返回其文本內容
def read_pdf(pdf_loc, page_infos=None):
    pdf = pdfplumber.open(pdf_loc)
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for page in pages:
        text = page.extract_text()
        if text:
            pdf_text += text
    pdf.close()
    return pdf_text

# 根據查詢語句和 `source` 限定檢索範圍
def retrieve_with_qdrant(query, source_ids, top_n=1):
    query_vector = model.encode(query).tolist()
    if len(query_vector) != embedding_dim:
        raise ValueError(f"查詢向量維度不匹配。期望的維度：{embedding_dim}，得到的維度：{len(query_vector)}")
    # 使用 `file_id` 過濾範圍檢索
    search_result = qdrant.search(
        collection_name="documents_collection",
        query_vector=query_vector,
        limit=top_n,
        filter={"key": "file_id", "match": {"value": source_ids}}
    )
    
    # 返回最相關的文件編號
    if search_result:
        return search_result[0].payload["file_id"]
    return None  # 如果無結果返回 None

# 主程式
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')
    args = parser.parse_args()

    answer_dict = {"answers": []}

    # 載入問題
    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)

    # 載入不同類別的參考資料
    corpus_dict_insurance = load_and_store_data(os.path.join(args.source_path, 'insurance'), "insurance")
    corpus_dict_finance = load_and_store_data(os.path.join(args.source_path, 'finance'), "finance")

    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
        for key, value in key_to_source_dict.items():
            vector = model.encode(value).tolist()
            qdrant.upsert(
                collection_name="documents_collection",
                points=[
                    PointStruct(
                        id=key,
                        vector=vector,
                        payload={"category": "faq", "file_id": key}
                    )
                ]
            )

    # 處理每個問題
    for q_dict in qs_ref['questions']:
        query = q_dict['query']
        category = q_dict['category']
        qid = q_dict['qid']
        source = q_dict['source']

        # 根據 `category` 檢索不同的資料庫
        if category in ["finance", "insurance", "faq"]:
            retrieved = retrieve_with_qdrant(query, source)
            answer_dict['answers'].append({
                "qid": qid,
                "retrieve": retrieved,
                "category": category
            })

    # 儲存檢索結果
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)
