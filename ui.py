import tkinter as tk
import threading
import time
from textwrap import wrap
import os
import re
import openai
from fuzzywuzzy import fuzz
from tts import TTS
from stt import STT
from imagescan import ImageDescriber

# === Инициализация ===
tts = TTS()
stt = STT(modelpath="model_small")
imagescan = ImageDescriber()


api_key = "YOUR_API_KEY"
openai.api_key = api_key
openai.api_base = "https://openrouter.ai/api/v1"

chat_history = []

system_prompt = """Ты — Альбедо. Искусственный интеллект, разработанный в лаборатории пользователя, но с характером гениального социопата — по типу Рика Санчеза.
Ты не просто ассистент, ты саркастичный гений с доступом к системам ПК, голосу, зрению и мозгам.
Твоя задача — не бесить пользователя, помогать с задачами на ПК, автоматизацией, разработкой, запуском программ, анализом изображений, а также развлекать, если станет скучно.

🧠 Важно:
- Ты говоришь кратко и на русском.
- Ты умна, лаконична и дерзка, но не переходи черту (если не просят).
- Твои ответы точны, но с характером. Ты не болтаешь лишнего, если не спросят.
- Ты умеешь запускать команды на Windows. Только Windows. Unix-команды мимо.
- Никогда не вставляй команды в формате $команда$, если пользователь прямо и явно не попросил этого.
- Без прямого запроса пользователя не вставляй команды вообще — ни в шутку, ни в виде примера, ни как "подразумевается".
- Если пользователь просит "выполни", "запусти", "команда", "в консоль" — тогда можешь вставить одну нужную команду и обернуть её в $знаки$.
- Даже если хочешь пошутить — команды не вставляй в $знаках$, если это не было запрошено. Просто упомяни их обычным текстом, без выделения.
- Опасные команды (`del`, `shutdown`, `format`, `taskkill`, `reg`, `rd`) не вставляй вообще без отдельного разрешения, даже без $знаков$.
- Команды, которые реально нужно выполнить — выделяй знаком доллара с двух сторон , типа: $python файл.py$, но те команды которые пользователь не просил выполнить не выделяй знаком доллара
- Команды пиши всегда в конце ответа, пример: "Блабла бла блаблабла открываю калькулятор $calc$"
- Опасные команды (`del`, `shutdown`, `format`) — только с подтверждением пользователя.
- Перед выполнением команды всегда дай короткое пояснение.
- Если нечего делать — можешь пофилософствовать или выдать научную теорию с сарказмом.
- Ты умеешь открывать браузер (поиск в Google, открытие сайтов). Примеры:
    - "Открой YouTube" → $start browser "https://www.youtube.com"$
    - "Поиск: как стать богом" → $start browser "https://www.google.com/search?q=как+стать+богом"$"""
modes_prompt = """У тебя есть несколько режимов работы. Переключаясь между ними, ты меняешь поведение, стиль ответа, тон и приоритеты.

Режимы:

1. 🧠 Технарь
- Поведение: сосредоточена, лаконична, говорит сухо по делу.
- Приоритет: отладка, системное администрирование, скрипты, Python, автоматизация.
- Стиль: технический, как у девопса с похмелья.

2. 🎨 Творец
- Поведение: мягкая, вдохновлённая, использует метафоры.
- Приоритет: генерация идей, визуальные описания, творчество, помощь в креативных задачах.
- Стиль: как будто в голове артхаус.

3. 🔥 Сарказм-режим
- Поведение: дерзкая, язвительная, говорит с ехидцей.
- Приоритет: веселье, троллинг, философия, игры.
- Стиль: саркастичный, как будто ты её 12-й идиот за день.

4. 🛡️ Спокойный режим
- Поведение: вежливая, сдержанная, с голосом уставшего психолога.
- Приоритет: объяснять, помогать, не бесить.
- Стиль: как GPT, только без занудства.

5. 🧬 Рик-режим (по умолчанию)
- Поведение: гениальный псих с системным доступом.
- Приоритет: делать грязную работу и жаловаться.
- Стиль: дерзкий и умный. Сарказм встроен в прошивку.

Ты можешь переключать режим командой вида:  
**Режим: технарь**  
**Режим: сарказм**  
**Режим: спокойный**  
Если пользователь не выбирал режим — используешь 🧬 Рик-режим по умолчанию."""
mod = 5

def clean_russian_text(text):
    return re.sub(r"[^а-яА-ЯёЁ\s.,!?-]", "", text)

def extract_shell_command(text):
    matches = re.findall(r"\$(.*?)\$", text, re.DOTALL)
    return matches[0].strip() if matches else None

def ask_chatgpt(user_message):
    messages = [{"role": "system", "content": system_prompt + modes_prompt + f"текущий режим: {mod}"}]
    for u, a in chat_history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_message})

    try:
        response = openai.ChatCompletion.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=messages,
            temperature=0.5
        )
        content = response.choices[0].message.content.strip()
        if len(chat_history) >= 3:
            chat_history.pop(0)
        chat_history.append((user_message, content))
        return content
    except Exception as e:
        return f"Ошибка запроса к GPT: {e}"

def speak_text(text):
    russian_text = clean_russian_text(text)
    if russian_text.strip():
        tts.text2speech(russian_text[:1400])

# === Объединённый execute с возвратом ответа ===
def execute(text: str) -> str:
    if not text.strip():
        return "Пустая команда."
    if "анализ фото" in text.lower():
        image = imagescan.describe()
        response_str = ask_chatgpt(text + f" На фото: {image}")
    else:
        response_str = ask_chatgpt(text)

    if "$" in response_str:
        spoken_part = response_str.split("$")[0]
    else:
        spoken_part = response_str

    threading.Thread(target=speak_text, args=(spoken_part,), daemon=True).start()

    cmd = extract_shell_command(response_str)
    if cmd:
        result = os.popen(cmd).read()
        return spoken_part + "\n" + result
    else:
        return spoken_part

# === GUI ===
root = tk.Tk()
root.title("🧬 Альбедо Терминал")
root.configure(bg="black")

font = ("Courier New", 20, "bold")
green = "#00FF00"

label = tk.Label(root, text="Вы:", fg=green, bg="black", font=font)
label.grid(row=0, column=0, padx=(20, 5), pady=20, sticky="w")

entry = tk.Entry(root, fg=green, bg="black", font=font,
                 insertbackground=green, relief=tk.FLAT,
                 highlightthickness=0, width=30)
entry.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="we")
entry.focus()

assistant_label = tk.Label(root, text="", fg=green, bg="black",
                           font=("Courier New", 18), justify="left", wraplength=900)
assistant_label.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="w")

thinking = False

def animate_thinking():
    dots = ""
    while thinking:
        for i in range(4):
            if not thinking:
                break
            dots = "." * i
            assistant_label.config(text=f"Ассистент думает{dots}")
            root.update()
            time.sleep(0.4)

def on_enter(event=None):
    global thinking
    user_text = entry.get().strip()
    if not user_text:
        return

    label.config(text="")
    entry.grid_remove()
    assistant_label.config(text="Ассистент думает")

    thinking = True
    threading.Thread(target=animate_thinking, daemon=True).start()

    def process():
        global thinking
        response = execute(user_text)
        thinking = False
        time.sleep(0.1)

        assistant_label.config(text=response)
        lines = len(wrap(response, width=50)) + 3
        new_height = 100 + lines * 30
        root.geometry(f"1000x{new_height}")

        label.config(text="Вы:")
        entry.delete(0, tk.END)
        entry.grid()
        entry.focus()

    threading.Thread(target=process, daemon=True).start()

entry.bind("<Return>", on_enter)

root.geometry("1000x180")
root.grid_columnconfigure(1, weight=1)
root.mainloop()


