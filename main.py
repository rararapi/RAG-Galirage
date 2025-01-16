import json
import sys
import time
import fitz
import shutil
import os
import re
import urllib.request
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain import callbacks
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
import warnings
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi

# ==============================================================================
# !!! 警告 !!!: 以下の変数を変更しないでください。
# ==============================================================================
model = "gpt-4o-mini"
pdf_file_urls = [
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Financial_Statements_2023.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Hada_Labo_Gokujun_Lotion_Overview.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Shibata_et_al_Research_Article.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/V_Rohto_Premium_Product_Information.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Well-Being_Report_2024.pdf",
]
# ==============================================================================


# ==============================================================================
# この関数を編集して、あなたの RAG パイプラインを実装してください。
# !!! 注意 !!!: デバッグ過程は標準出力に出力しないでください。
# ==============================================================================
def rag_implementation(question: str) -> str:
    """
    複数のチャンクサイズでRAGパイプラインを実行し、回答を統合する関数

    Args:
        question (str): 質問文字列

    Returns:
        answer (str): 統合された最終的な回答
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.basicConfig(level=logging.ERROR)

    def load_pdf_with_pymupdf(url):
        """URLからPDFをダウンロードしてテキストを抽出"""
        response = urllib.request.urlopen(url)
        pdf_data = response.read()
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        texts = []
        for page in pdf_document:
            texts.append(page.get_text())
        pdf_document.close()
        return "\n".join(texts)

    def download_and_load_pdfs(urls):
        """URLからPDFをダウンロードし、pymupdfでテキストを抽出"""
        documents = []
        for url in urls:
            try:
                # PDFテキストを抽出
                text_content = load_pdf_with_pymupdf(url)
                cleaned_text = normalize_text(text_content)

                documents.append({"page_content": cleaned_text, "metadata": {"source": url}})
            except Exception as e:
                pass
        return documents

    def normalize_text(s):
        s = re.sub(r'\s+', ' ', s)  # 連続する空白を1つに
        s = s.replace("..", ".").replace(". .", ".")
        s = s.strip()
        return s

    def split_documents(documents, chunk_size=800, chunk_overlap=400):
        """長いテキストを分割する。空のドキュメントをスキップ"""
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = []
        for doc in documents:
            if doc["page_content"].strip():  # 空でない場合のみ処理
                # CharacterTextSplitter は docstrings を返すが、ここでは文字列だけで運用
                chunks = text_splitter.split_text(doc["page_content"])
                for chunk in chunks:
                    if chunk.strip():
                        # chunk を "page_content" に格納する形で取り扱う
                        split_docs.append({
                            "page_content": chunk,
                            "metadata": doc["metadata"]
                        })
        return split_docs

    def filter_documents_by_bm25(question, docs, top_k=50):
        """
        質問に対してBM25でスコアリングし、上位 top_k 件の文書だけを返す
        docs: [{"page_content": str, "metadata": dict}, ...]
        """
        # BM25は「トークン化された単語リスト」でスコアリングするため、簡易的にsplit
        corpus = [d["page_content"] for d in docs]
        tokenized_corpus = [c.split() for c in corpus]

        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = question.split()
        scores = bm25.get_scores(tokenized_query)

        # スコアの高い順にソートして上位 top_k を抽出
        scored_docs = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        filtered_docs = [x[0] for x in scored_docs]
        return filtered_docs

    def construct_few_shot_prompt(question):
        """
        Few-shotプロンプトを構築する関数
        """
        few_shot_examples = [
            {
                "question": "存在意義（パーパス）は、なんですか？",
                "answer": "世界の人々に商品やサービスを通じて「健康」をお届けすることによって、当社を取り巻くすべての人や社会を「Well-being」へと導き、明日の世界を元気にすることです。",
            },
            {
                "question": "事務連絡者の電話番号は？",
                "answer": "（06）6758-1235です。",
            },
            {
                "question": "Vロートプレミアムは、第何類の医薬品ですか？",
                "answer": "第2類医薬品です。",
            },
            {
                "question": "肌ラボ 極潤ヒアルロン液の詰め替え用には、何mLが入っていますか？",
                "answer": "170mLが入っています。",
            },
            {
                "question": "LN211E8は、どのようなhiPSCの分化において、どのように作用しますか？",
                "answer": "Wnt 活性化を通じて神経堤細胞への分化を促進します。",
            },
        ]

        # Few-shotプロンプトを構築
        prompt = "以下は質問と回答の例です。\n\n"
        for example in few_shot_examples:
            prompt += f"質問: {example['question']}\n{example['answer']}\n\n"

        # 最後にユーザーの質問を追加
        prompt += f"質問: {question}\n"
        return prompt


    try:
        # 1. PDFをダウンロードしてまとめてテキスト抽出
        documents = download_and_load_pdfs(pdf_file_urls)

        # 2. 複数のチャンクサイズでドキュメントを分割
        chunk_configs = [
            {"chunk_size": 500, "chunk_overlap": 200},
            {"chunk_size": 800, "chunk_overlap": 400},
            {"chunk_size": 1100, "chunk_overlap": 600},
        ]

        answer_candidates = []

        for config in chunk_configs:
            # ドキュメントを分割
            split_docs = split_documents(documents, chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"])
            
            # 質問との関連度が高いチャンクを抽出
            filtered_docs = filter_documents_by_bm25(question, split_docs, top_k=3)

            # ベクトルストアを作成
            embeddings = OpenAIEmbeddings()
            vector_store_dir = f"DB_store_{config['chunk_size']}"
            if os.path.exists(vector_store_dir):
                shutil.rmtree(vector_store_dir)  # 以前のストアを削除

            vector_store = Chroma.from_texts(
                texts=[doc["page_content"] for doc in filtered_docs],
                embedding=embeddings,
                metadatas=[doc["metadata"] for doc in filtered_docs],
                persist_directory=vector_store_dir
            )

            # RetrievalQAチェーンを実行
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model=model),
                retriever=retriever,
                chain_type="stuff"
            )

            # 各チャンクサイズに基づく回答を取得
            answer_candidate = qa_chain.run(question)
            answer_candidates.append(answer_candidate)
        

        # Few-shot プロンプトを使って最終回答を統合
        few_shot_prompt = construct_few_shot_prompt(question)
        final_llm = ChatOpenAI(model=model)
        final_prompt = [
            SystemMessage(content=(
                "以下は質問とその回答例です。参考にして、与えられた質問に適切な回答を短い一文で答えてください。\n"
                "与えられた情報が不足している場合でも、背景情報を補足し、最も妥当な回答を推測してください。\n"
                "以下のステップを参考に回答を作成してください：\n"
                "1. 質問の主旨を正確に理解する。\n"
                "2. 回答候補を比較し、最も関連性の高いものを選択する。\n"
                "3. 簡潔かつ正確な回答を記述する。\n\n"
                f"{few_shot_prompt}"
            )),
            HumanMessage(content=(
                f"質問: {question}\n\n"
                "以下は製薬企業のドキュメントから抽出された回答候補です。\n"
                f"回答候補:\n{answer_candidates}\n\n"
                "これらを踏まえて、最も適切な回答を簡潔に記述してください。\n"
                "Let's think step by step."
            )),
        ]
        final_answer = final_llm.invoke(final_prompt).content

        return final_answer

    except Exception as e:
        return "資料から回答することができませんでした。"


# ==============================================================================
# !!! 警告 !!!: 以下の関数を編集しないでください。
# ==============================================================================
def main(question: str):
    with callbacks.collect_runs() as cb:
        result = rag_implementation(question)

        for attempt in range(2):  # 最大2回試行
            try:
                run_id = cb.traced_runs[0].id
                break
            except IndexError:
                if attempt == 0:  # 1回目の失敗時のみ
                    time.sleep(3)  # 3秒待機して再試行
                else:  # 2回目も失敗した場合
                    raise RuntimeError("Failed to get run_id after 2 attempts")

    output = {"result": result, "run_id": str(run_id)}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    load_dotenv()

    if len(sys.argv) > 1:
        question = sys.argv[1]
        main(question)
    else:
        print("Please provide a question as a command-line argument.")
        sys.exit(1)
