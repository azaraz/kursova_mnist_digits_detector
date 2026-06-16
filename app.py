import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

model = tf.keras.models.load_model('mnist_cnn.keras')


def preprocess_digit(image) :
    image = image.convert('L')

    if np.mean(image) > 127 :
        image = ImageOps.invert(image)

    bbox = image.getbbox()
    if bbox is None :
        return None

    cropped_image = image.crop(bbox)

    max_size = 20
    ratio = max_size / max(cropped_image.size)
    new_size = (int(cropped_image.size[0] * ratio), int(cropped_image.size[1] * ratio))
    resized_image = cropped_image.resize(new_size, Image.Resampling.LANCZOS)

    final_image = Image.new("L", (28, 28), 0)

    x = (28 - new_size[0]) // 2
    y = (28 - new_size[1]) // 2
    final_image.paste(resized_image, (x, y))

    return final_image

def predict_digit(sketch) :
    try :
        if isinstance(sketch, dict) :
            img_array = sketch.get('composite', sketch.get('background'))
        else :
            img_array = sketch

        if img_array is None :
            return {"Помилка: Намалюйте цифру" : 0.0}

        raw_image = Image.fromarray(img_array)

        processed_image = preprocess_digit(raw_image)

        if processed_image is None :
            return {"Помилка: Полотно пусте" : 0.0}

        img_data = np.array(processed_image).reshape(1, 28, 28, 1) / 255.0

        prediction = model.predict(img_data)[0]

        confidences = {str(i) : float(prediction[i]) for i in range(10)}

        return confidences

    except Exception as e :
        return {f"Помилка: {str(e)}" : 0.0}


interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Намалюйте цифру від 0 до 9", type="numpy"),
    outputs=gr.Label(num_top_classes=3, label="Результат розпізнавання"),
    title="Розпізнавання рукописних цифр"
)

if __name__ == "__main__" :
    interface.launch()