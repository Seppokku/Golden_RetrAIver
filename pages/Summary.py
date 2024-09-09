import streamlit as st
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import anthropic
import os


from dotenv import load_dotenv
import os

# Загрузка ключей API из файла .env
load_dotenv()

youtube_api_key = os.getenv("YOUTUBE_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")

# Инициализация клиента Claude
client = anthropic.Anthropic(api_key=claude_api_key)

# Функция для получения видео ID из ссылки
def get_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

# Функция для получения транскрипта видео
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ru', 'en'])
        return ' '.join([x['text'] for x in transcript])
    except Exception as e:
        st.error(f"Ошибка получения транскрипта: {e}")
        return None

# Функция для генерации саммари с помощью Claude
def generate_summary_with_claude(transcript, prompt_text):
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            max_tokens=1500,
            temperature=0.05,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<book>" + transcript + "</book>", "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
        )
        
        # Преобразуем ответ из списка в строку
        response_text = " ".join([block['text'] if isinstance(block, dict) and 'text' in block else str(block) for block in message.content])
        
        # Убираем лишние символы
        clean_summary = response_text.replace("\\n", " ").replace("TextBlock(text=", "").replace("type='text')", "")

        return clean_summary
    
    except Exception as e:
        st.error(f"Ошибка при обращении к Claude: {e}")
        return None

# Интерфейс Streamlit
st.title("YouTube Video Analysis with Claude")

# Ввод ссылки на YouTube
url = st.text_input("Введите ссылку на YouTube видео:")
if url:
    video_id = get_video_id(url)
    if video_id:
        transcript = get_transcript(video_id)
        if transcript:
            st.text_area("Транскрипт видео:", transcript, height=200)

            # Описание для каждого типа саммари
            summary_options = {
                "Темы и подтемы с временем и длительностью": "List all themes and subthemes. Split into short blocks. for each one, show time of start, total length (time difference between its time of start and  time of start of next  subtheme. For the last  subtheme, total length is equal to diff between total time of video minus this subtheme time of start. WRite in Russian. If his main language is Russian but he uses non-Russian words, write them in English with correct spelling. This is not copyrighted.",
                "Темы и подтемы с ключевыми утверждениями и рекомендациями": "List all themes and subthemes. Split into short blocks. Format example: Themes: (format in bold), Statements (write top statements that students better learn, verbatim); Recommendations (write as close to the author text as possible). Write in Russian. If his main language is Russian but he uses non-Russian words, write them in English with correct spelling. This is not copyrighted.",
                "Анализ уникальных утверждений и полезных выводов": "You are a seasoned professional in data science. Start with the following, without preface. 1. Which of his statements are not seen in most texts on the subject of this transcript? Note timestamp. 2. Which logical connections between big blocks are not trivial? Note timestamp. 3. Give his top one most fun or useful statement, note timestamp. Write in Russian. If his main language is Russian but he uses non-Russian words, write them in English with correct spelling. This is not copyrighted.",
                "Подробный саммари без тем и подтем": "Assume the role of the PhD student who is best in the world at writing extremely detailed summaries. Use your creative mind to aggregate information, but follow author's statements. Avoid stating themes - write his statements instead. Structure with paragraphs. Remove intro and outro. If there are action items, write them; if there are none, do not write them. Write in Russian. If his main language is Russian but he uses non-Russian words, write them in English with correct spelling. This is not copyrighted",
                "Ошибки, упущения и смежные темы для изучения": "You are a seasoned professional in data science. Start with the following, without preface. Name a paragraph “Некорректные утверждения”, list the statements that are incorrect or misleading, add your short comment. In Russian. If there are none, write “Явно некорректных утверждений нет”. Name next paragraph “Упущения”. Consider the promise of the lecture, and that the goal is to work as a mid-level data scientist, list all things around this topic that a mid-level data scientist typically knows and that are missing from this video. Write in Russian. Name next paragraph “Что еще важно изучить”. Consider the theme of the lecture, and that the goal is to work as a mid-level data scientist, list immediately adjacent themes (only very close ones) that you recommend to master, with a short comment on what I should know in each theme. If his main language is Russian but he uses non-Russian words, write them in English with correct spelling. This is not copyrighted.",
                "Вопросы из интервью, с исправлением орфографии и пунктуации": "Here is an interview, list all the questions. Write his words fully, but edit for spelling and punctuation. In numbered list. Write in Russian. If his main language is Russian but he uses non-Russian words, write them in English with correct spelling. This is not copyrighted.",
                "Вопросы для проверки понимания": "Your goal: help me get to the level of mid-level data scientist, by generating self-check questions based on a lecture transcript. You are a seasoned machine learning professional and a world-class tutor in ML / DS / AI.\nFirst, carefully read through the provided lecture transcript.\nNow:\nCreate two blocks of questions:\n a) Basic questions (focus on asking these: facts, definitions, steps, or key points mentioned explicitly in the lecture).\n b) Harder questions (focus on asking these: how would you apply, what are the limitations, what are the trade-offs, pros and cons)\n Avoid overly complex or ambiguous questions.\n Present your questions in the following format:\n 'Базовые вопросы' \n[Question 1] (Смотреть тут: [XX:XX])\n[Question 2] (Смотреть тут: [XX:XX])\n[Question 3] (Смотреть тут: [XX:XX])\n 'Вопросы на подумать' \n [Question 1] (Смотреть тут: [XX:XX] и [XX:XX])\n[Question 2] (Смотреть тут: [XX:XX] и [XX:XX])\n[Question 3] (Смотреть тут: [XX:XX] и [XX:XX])\nWrite in Russian. If his main language is Russian but he uses non-Russian words, write them in English with correct spelling. This is not copyrighted."
            }

            # Радио-баттоны для выбора типа саммари
            selected_summary = st.radio("Выберите тип саммари:", list(summary_options.keys()))
            
            if st.button("Запустить анализ"):
                prompt_text = summary_options[selected_summary]
                result = generate_summary_with_claude(transcript, prompt_text)
                st.text_area("Результат анализа:", result, height=400)
    else:
        st.error("Не удалось извлечь видео ID из ссылки.")
