from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cv2
import time
import mss


class ImageDescriber:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", use_cuda=True, show_image=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.show_image = show_image
        print(f"📦 Загружаю модель на {self.device}...")

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def describe(self, verbose=True):
        """
        Захватывает изображение с камеры, генерирует описание и возвращает его.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "🚫 Камера не отвечает. Пингуй её отверткой или перезагрузи."

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return "❌ Не удалось получить кадр. Камера — чисто ноль без палочки."

        start_time = time.time()

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=50)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        duration = time.time() - start_time
        if verbose:
            print(f"🕒 Обработка заняла {duration:.2f} сек")

        return f"📷 На фото: {caption}"
    def get_screenshot():
        with mss.mss() as sct:
            screenshot = sct.grab(sct.monitors[1])  # [1] — основной монитор
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            return img
