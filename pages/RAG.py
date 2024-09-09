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
embedding_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

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
            max_tokens=300,
            temperature=0.7
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"Ошибка при вызове модели: {e}")
        return None

# Функция для генерации ответа на вопрос пользователя
def answer_question(question, retriever, client):
    # Получение релевантных документов из базы знаний
    documents = retriever.get_relevant_documents(question)
    context = " ".join([doc.page_content for doc in documents])

    # Формирование запроса к модели
    prompt = prompt_template.format(context=context, input=question)

    # Вызов API модели Claude
    answer = call_claude_api(prompt, client)
    return answer, documents


st.title("Поиск по базе знаний RAG с моделью Claude")

st.write("Используйте базу знаний для поиска информации и генерации ответов.")

# Поле для ввода запроса пользователя
query = st.text_input("Введите ваш запрос:", 'Что такое машинное обучение?')

if st.button("Поиск и генерация ответа"):
    if query:

        # Генерация ответа на вопрос
        answer, documents = answer_question(query, embedding_retriever, client)

        if answer:
            # Отображение ответа
            st.text_area("Ответ:", answer, height=150)

            # Отображение контекста (фрагменты из базы знаний)
            st.subheader('Контекст')
            for i, doc in enumerate(documents):
                st.subheader(f'Фрагмент {i+1}')
                st.write(doc.page_content)
        else:
            st.warning("Не удалось получить ответ от модели.")
    else:
        st.warning("Пожалуйста, введите запрос.")




