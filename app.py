import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Завантажуємо навчену модель
model = tf.keras.models.load_model('mnist_cnn.keras')


def preprocess_digit(image) :
    """
    Функція для підготовки зображення.
    Імітує оригінальний алгоритм створення датасету MNIST.
    """
    # 1. Переводимо у чорно-білий формат
    image = image.convert('L')

    # 2. Інвертуємо кольори (якщо фон білий, робимо його чорним)
    if np.mean(image) > 127 :
        image = ImageOps.invert(image)

    # 3. Знаходимо межі самої цифри (Bounding Box) і відрізаємо порожнечу
    bbox = image.getbbox()
    if bbox is None :
        return None  # Якщо користувач нічого не намалював

    cropped_image = image.crop(bbox)

    # 4. Змінюємо розмір так, щоб найбільша сторона була 20 пікселів (Стандарт MNIST)
    # Це зберігає правильні пропорції цифри
    max_size = 20
    ratio = max_size / max(cropped_image.size)
    new_size = (int(cropped_image.size[0] * ratio), int(cropped_image.size[1] * ratio))
    resized_image = cropped_image.resize(new_size, Image.Resampling.LANCZOS)

    # 5. Створюємо нове чорне полотно 28x28
    final_image = Image.new("L", (28, 28), 0)  # 0 - це чорний колір

    # 6. Вставляємо нашу цифру рівно по центру
    x = (28 - new_size[0]) // 2
    y = (28 - new_size[1]) // 2
    final_image.paste(resized_image, (x, y))

    return final_image

def predict_digit(sketch) :
    try :
        # Отримуємо малюнок з інтерфейсу Gradio
        if isinstance(sketch, dict) :
            img_array = sketch.get('composite', sketch.get('background'))
        else :
            img_array = sketch

        if img_array is None :
            return {"Помилка: Намалюйте цифру" : 0.0}

        # Перетворюємо масив у зображення
        raw_image = Image.fromarray(img_array)

        # ЗАПУСКАЄМО НАШУ НОВУ СУПЕР-ФУНКЦІЮ ОЧИСТКИ
        processed_image = preprocess_digit(raw_image)

        if processed_image is None :
            return {"Помилка: Полотно пусте" : 0.0}

        # Перетворюємо назад у масив, нормалізуємо та додаємо необхідні виміри
        img_data = np.array(processed_image).reshape(1, 28, 28, 1) / 255.0

        # Робимо прогноз
        prediction = model.predict(img_data)[0]

        # Створюємо словник результатів: {"0": 0.01, "1": 0.98, ...}
        confidences = {str(i) : float(prediction[i]) for i in range(10)}

        return confidences

    except Exception as e :
        return {f"Помилка: {str(e)}" : 0.0}


# Створюємо інтерфейс з полотном для малювання
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Намалюйте цифру від 0 до 9", type="numpy"),
    outputs=gr.Label(num_top_classes=3, label="Результат розпізнавання"),
    title="Розпізнавання рукописних цифр"
)

if __name__ == "__main__" :
    interface.launch()