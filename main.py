import json
import sys
import time
import fitz
import shutil
import os
import tempfile
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
            text_content = load_pdf_with_pymupdf(url)
            documents.append({"page_content": text_content, "metadata": {"source": url}})
        except Exception as e:
            print(f"Failed to process PDF from {url}: {e}")
    return documents

def split_documents(documents, chunk_size=80, chunk_overlap=20):
    """長いテキストを分割する。空のドキュメントをスキップ"""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = []
    for doc in documents:
        if doc["page_content"].strip():  # 空でない場合のみ処理
            split_docs.extend(text_splitter.split_text(doc["page_content"]))
    return split_docs


def cleanup_old_batches(n):
    """既存のDB_batch_ディレクトリを削除"""
    for idx in range(n):  # 最大10バッチを想定
        batch_dir = f"DB_batch_{idx}"
        if os.path.exists(batch_dir):
            shutil.rmtree(batch_dir)

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

def rag_implementation(question: str) -> str:
    """
    ロート製薬の製品・企業情報に関する質問に対して回答を生成する関数
    この関数は与えられた質問に対してRAGパイプラインを用いて回答を生成します。

    Args:
        question (str): ロート製薬の製品・企業情報に関する質問文字列

    Returns:
        answer (str): 質問に対する回答

    Note:
        - デバッグ出力は標準出力に出力しないでください
        - model 変数 と pdf_file_urls 変数は編集しないでください
        - 回答は日本語で生成してください
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.basicConfig(level=logging.ERROR)

    try:
        # URLからPDFをダウンロードしてロード
        documents = download_and_load_pdfs(pdf_file_urls)

        # ドキュメントを分割
        split_docs = split_documents(documents)

        batch_size = 4
        # 古いバッチディレクトリをクリーンアップ
        cleanup_old_batches(batch_size)
        # 分割ドキュメントをバッチ化
        batched_docs = [split_docs[i::batch_size] for i in range(batch_size)]

        # 各バッチの回答を取得
        intermediate_answers = []
        embeddings = OpenAIEmbeddings()
        
        for idx, batch in enumerate(batched_docs):
            vector_store = Chroma.from_texts(batch, embeddings, persist_directory=f"DB_batch_{idx}")
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model=model),
                retriever=retriever,
                chain_type="stuff"
            )
            try:
                answer = qa_chain.run(question)
                intermediate_answers.append(answer)
            except Exception as e:
                print(f"Error processing batch {idx + 1}: {e}")
                intermediate_answers.append("")

        # 最終回答を統合
        combined_answer = "\n".join(intermediate_answers)
        try:
            # Few-shotプロンプトを構築
            few_shot_prompt = construct_few_shot_prompt(question)

            # `final_prompt`を定義してLLMに渡す
            final_llm = ChatOpenAI(model=model)
            final_prompt = [
                # Few-shotプロンプトを背景情報として提供
                SystemMessage(content=(
                    "以下は質問とその回答例です。参考にして、与えられた質問に適切な回答を生成してください。\n\n"
                    f"{few_shot_prompt}"
                )),
                # ユーザーの質問と回答候補をHumanMessageで提供
                HumanMessage(content=(
                    f"質問: {question}\n\n"
                    "以下はバッチ処理されたドキュメントから生成された回答候補です。\n"
                    "回答候補:\n" +
                    "\n".join([f"- {ans}" for ans in combined_answer.split('\n') if ans.strip()]) +
                    "\n\nこれらの情報を基に、質問に対する最も適切な回答を記述してください。"
                )),
            ]
            final_answer = final_llm.invoke(final_prompt).content
        except Exception as e:
            final_answer = "資料から回答することができませんでした。"

        return final_answer
    except Exception as e:
        return "資料から回答することができませんでした。"


# ==============================================================================


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
# ==============================================================================
