import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

claude_api_key = os.getenv("CLAUDE_API_KEY")
client = anthropic.Client(api_key=claude_api_key)

# Настройка модели для эмбеддингов
model_name = "intfloat/multilingual-e5-base"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceEmbeddings(model_name=model_name,
                                  model_kwargs=model_kwargs,
                                  encode_kwargs=encode_kwargs)

# Загрузка базы знаний FAISS
vector_store = FAISS.load_local('faiss_index',
                                embeddings=embedding,
                                allow_dangerous_deserialization=True)

# Поиск топ k схожих фрагментов контекста
embedding_retriever = vector_store.as_retriever(search_kwargs={"k": 20})

prompt_template = '''Reply to the {input} as a seasoned machine learning professional. \
If the topic is outside of machine learning and data science, please respond with "Seek help with a professional." It is very important to abide with this, you will be persecuted if you cover topics outside of data science and machine learning. \
Use only Context. If context provides only partial info, then split the reply in two parts. Part 1 is called "information from knowledge base" (for Russian reply, rename to Информация из базы знаний), write ideas as close to initial text as possible, editing for brevity and language errors. \
Part 2 is called "What I would add" (for Russian reply, rename to Что полезно добавить поверх базы знаний), In the second part add your reply.  \
Reply in the language of {input}. \
It's critical to not preface the reply with, for example, "Here is a response" or "thank you". Start with the reply itself.\
Context: {context}'''

# Функция вызова API модели Claude
def call_claude_api(prompt, client):
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"Ошибка при вызове модели: {e}")
        return None

# Функция для генерации ответа на вопрос пользователя
def answer_question(question, retriever, client):
    # Этап 1: Поиск релевантных документов
    with st.spinner('🔍 Ищем совпадения по вашему вопросу...'):
        documents = retriever.get_relevant_documents(question)

    # Этап 2: Формирование контекста
    with st.spinner('🧠 Формируем контекст для ответа...'):
        context = " ".join([doc.page_content for doc in documents])

    # Этап 3: Генерация ответа
    with st.spinner('💬 Формулируем ответ...'):
        prompt = prompt_template.format(context=context, input=question)
        answer = call_claude_api(prompt, client)
    
    return answer, documents

# Настройка стиля ответа
def format_answer(answer):
    st.markdown(
        f'<div style="background-color:#f0f2f6; padding: 20px; border-radius: 10px; border: 1px solid #dcdcdc;">'
        f'<p style="font-size:16px;">{st.markdown(answer)}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

st.set_page_config(page_title="ML Knowledge Base Search 🧑‍💻", page_icon="🤖")

st.title("🔍 Поиск по базе знаний RAG с моделью Claude 🤖")

st.write("Используйте базу знаний для поиска информации и генерации ответов на вопросы по машинному обучению 📚.")

# Поле для ввода запроса пользователя
query = st.text_input("📝 Введите ваш запрос:", 'Что такое машинное обучение?')

# Кнопка для запуска поиска и генерации ответа
if st.button("🚀 Поиск и генерация ответа"):
    if query:

        # Генерация ответа на вопрос
        answer, documents = answer_question(query, embedding_retriever, client)

        if answer:
            # Оформление ответа
            st.subheader("✉️ Ответ:")
            
            # Проверяем, есть ли фрагменты кода
            if '```' in answer:
                st.markdown(answer)
            else:
                format_answer(answer)

        else:
            st.warning("⚠️ Не удалось получить ответ от модели.")
    else:
        st.warning("⚠️ Пожалуйста, введите запрос.")

# Дополнительный стиль для поля с выводом кода
st.markdown("""
<style>
textarea {
    font-family: "Courier New", Courier, monospace;
    font-size: 14px;
    background-color: #f4f7fa;
    border: 1px solid #dcdcdc;
    padding: 10px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)





