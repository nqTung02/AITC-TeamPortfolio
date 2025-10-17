import os
import json
import dotenv
import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Thiếu OpenAI API key. Vui lòng thêm vào file .env.")

IMAGE_FOLDER = "dataset/images"
CAPTION_FILE = "image_captions.json"

with open(CAPTION_FILE, "r", encoding="utf-8") as f:
    image_captions = json.load(f)

vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini")

embed_model = SentenceTransformer("keepitreal/vietnamese-sbert")

def embed(text):
    return embed_model.encode(text)

answer_prompt = PromptTemplate.from_template(
    """Bạn là một trợ lý AI hỗ trợ trả lời các câu hỏi bằng tiếng Việt, dựa trên nội dung được cung cấp bên dưới.

Nội dung tham khảo:
{context}

Câu hỏi:
{question}

Hãy trả lời một cách chi tiết, rõ ràng và dễ hiểu. Nếu có thể, hãy giải thích thêm và đưa ra ví dụ minh họa.

Trả lời:"""
)

summarize_prompt = PromptTemplate.from_template(
    """Hãy viết một đoạn tóm tắt 1 câu ngắn gọn bằng tiếng Việt cho nội dung sau.

{context}

Tóm tắt:"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | answer_prompt
    | llm
    | StrOutputParser()
)

summarize_chain = (
    {"context": RunnablePassthrough()}
    | summarize_prompt
    | llm
    | StrOutputParser()
)

def find_related_images(summary_text, image_folder=IMAGE_FOLDER, top_k=2, threshold=0.5):
    answer_vec = embed(summary_text)
    matches = []

    print(f"\n[DEBUG] Tóm tắt: {summary_text}")
    print("[DEBUG] Đang so sánh với các caption:")

    for img_file, caption in image_captions.items():
        full_path = os.path.join(image_folder, img_file)
        if not os.path.isfile(full_path):
            print(f"Bỏ qua: {img_file} (file không tồn tại)")
            continue

        caption_vec = embed(caption)
        sim = cosine_similarity([answer_vec], [caption_vec])[0][0]
        status = "Giữ lại" if sim >= threshold else "Bỏ qua"
        print(f"{img_file}: \"{caption}\" → Similarity: {sim:.4f} {status}")

        if sim >= threshold:
            matches.append((full_path, sim))

    matches.sort(key=lambda x: x[1], reverse=True)
    print("\n🏁 [DEBUG] Top ảnh được chọn:")
    for path, score in matches[:top_k]:
        print(f"  → {os.path.basename(path)} (score: {score:.4f})")

    return [m[0] for m in matches[:top_k]]

def answer_question(question):
    if not question.strip():
        return "Vui lòng nhập câu hỏi.", "", None, None

    try:
        non_rag_answer = llm.invoke(question).content
    except Exception as e:
        non_rag_answer = f"Lỗi GPT: {str(e)}"

    try:
        rag_answer = rag_chain.invoke(question)
        summary = summarize_chain.invoke(rag_answer)
        print("Tóm tắt:", summary)
        image_paths = find_related_images(summary)
    except Exception as e:
        rag_answer = f"Lỗi RAG: {str(e)}"
        image_paths = []

    images = [None] * 2
    for i in range(min(len(image_paths), 2)):
        images[i] = image_paths[i]

    return non_rag_answer, rag_answer, *images

with gr.Blocks() as demo:
    gr.Markdown("## Chatbot AOF")

    question_input = gr.Textbox(label="Nhập câu hỏi bằng tiếng Việt")
    btn = gr.Button("Lấy câu trả lời")

    non_rag_output = gr.Textbox(label="Câu trả lời của LLM/GPT", lines=4)
    rag_output = gr.Textbox(label="Câu trả lời của AOF Chatbot", lines=8)

    gr.Markdown("### Hình ảnh liên quan")
    with gr.Row():
        image1 = gr.Image(label="Ảnh 1", type="filepath")
        image2 = gr.Image(label="Ảnh 2", type="filepath")

    btn.click(
        fn=answer_question,
        inputs=question_input,
        outputs=[
            non_rag_output, rag_output,
            image1, image2
        ]
    )

demo.launch(share=True)
