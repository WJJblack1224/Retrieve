import os
import json
import argparse
from tqdm import tqdm
import jieba
import pdfplumber
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 初始化模型
model = SentenceTransformer('shibing624/text2vec-base-chinese')

# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path):
    masked_file_ls = os.listdir(source_path)
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}
    return corpus_dict

# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos=None):
    pdf = pdfplumber.open(pdf_loc)
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''.join([page.extract_text() or '' for page in pages])
    pdf.close()
    return pdf_text

# 混合檢索方法
def hybrid_retrieve(query, source, corpus_dict, top_k_stf=3):
    # 1. 使用 SentenceTransformer 選出前 top_k_stf 個文件
    query_vector = model.encode(query).reshape(1, -1)
    candidate_docs = []
    for file_id in source:
        doc_text = corpus_dict.get(int(file_id), "")
        text_vector = model.encode(doc_text).reshape(1, -1)
        score = cosine_similarity(query_vector, text_vector)[0][0]
        candidate_docs.append((file_id, doc_text, score))
    candidate_docs = sorted(candidate_docs, key=lambda x: x[2], reverse=True)[:top_k_stf]

    # 2. 在候選文件中使用 BM25 進行關鍵字檢索
    tokenized_corpus = [list(jieba.cut_for_search(doc[1])) for doc in candidate_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(jieba.cut_for_search(query))
    bm25_top_doc = bm25.get_top_n(tokenized_query, candidate_docs, n=1)[0]  # 只取最相關的文件

    return bm25_top_doc[0]  # 返回最相關文件的 ID

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')
    args = parser.parse_args()

    answer_dict = {"answers": []}

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)

    source_path_insurance = os.path.join(args.source_path, 'insurance')
    corpus_dict_insurance = load_data(source_path_insurance)

    source_path_finance = os.path.join(args.source_path, 'finance')
    corpus_dict_finance = load_data(source_path_finance)

    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)
        corpus_dict_faq = {int(key): str(value) for key, value in key_to_source_dict.items()}

    for q_dict in qs_ref['questions']:
        query = q_dict['query']
        category = q_dict['category']
        qid = q_dict['qid']
        source = q_dict['source']

        # 選擇資料來源並進行混合檢索
        if category == 'finance':
            retrieved = hybrid_retrieve(query, source, corpus_dict_finance)
        elif category == 'insurance':
            retrieved = hybrid_retrieve(query, source, corpus_dict_insurance)
        elif category == 'faq':
            filtered_corpus_faq = {key: value for key, value in corpus_dict_faq.items() if key in source}
            retrieved = hybrid_retrieve(query, source, filtered_corpus_faq)
        else:
            raise ValueError("Invalid category")

        answer_dict['answers'].append({
            "qid": qid,
            "retrieve": retrieved,
            "category": category
        })

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)

    print("Retrieval completed and saved successfully.")
