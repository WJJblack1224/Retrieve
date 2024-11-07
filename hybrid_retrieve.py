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
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

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
def hybrid_retrieve(qs, source, corpus_dict, top_k_bm25=5):
    # 1. 使用 BM25 進行初步篩選
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(jieba.cut_for_search(qs))
    bm25_top_docs = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=top_k_bm25)

    # 2. 使用 SentenceTransformer 計算語意相似度
    query_vector = model.encode(qs).reshape(1, -1)
    best_match_id, best_score = None, -1
    for text in bm25_top_docs:
        for file_id, doc_text in corpus_dict.items():
            if doc_text == text:
                text_vector = model.encode(text).reshape(1, -1)
                score = cosine_similarity(query_vector, text_vector)[0][0]
                if score > best_score:
                    best_score = score
                    best_match_id = file_id

    return best_match_id

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
