from pathlib import Path
import sys

import numpy as np
from PIL import Image
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_mobilenetv2_model.keras"
IMG_SIZE = (160, 160)
CLASS_NAMES = ["cat", "dog"]


def load_image(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(image_path: str) -> None:
    image_path = Path(image_path)

    if not image_path.exists():
        print("Файл не знайдено")
        return

    model = tf.keras.models.load_model(
        MODEL_PATH,
        safe_mode=False,
        custom_objects={"preprocess_input": preprocess_input}
    )

    img_array = load_image(image_path)
    prediction = model.predict(img_array, verbose=0)[0][0]

    predicted_class = CLASS_NAMES[int(prediction >= 0.5)]
    confidence = prediction if predicted_class == "dog" else 1 - prediction

    print(f"Результат: {predicted_class}")
    print(f"Впевненість: {confidence:.2%}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Використання: python src/predict.py шлях_до_зображення")
    else:
        predict_image(sys.argv[1])