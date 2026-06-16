import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("1. Завантаження даних MNIST...")
# Датасет вже вбудований у Keras, його не треба качати руками!
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print("2. Підготовка даних...")
# Нормалізація (перетворюємо яскравість пікселів з діапазону 0-255 у 0.0-1.0)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Зміна форми для CNN (додаємо 1 канал кольору, оскільки зображення чорно-білі)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print("3. Створення CNN (Згорткової нейромережі)...")
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax') # 10 класів на виході (цифри від 0 до 9)
])

# Компіляція моделі
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("4. Навчання моделі (це займе близько хвилини)...")
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=64)

print("5. Збереження моделі...")
model.save('mnist_cnn.keras')

print("6. Збереження графіків для звіту...")
plt.plot(history.history['accuracy'], label='Тренування')
plt.plot(history.history['val_accuracy'], label='Тест (Валідація)')
plt.title('Точність навчання моделі (Accuracy)')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()
plt.savefig('mnist_accuracy.png')
plt.close()

print("\n--- ОЦІНКА ЯКОСТІ (ДЛЯ РОЗДІЛУ 5 У ЗВІТІ) ---")
# Робимо передбачення для тестової вибірки
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Для багатокласової класифікації використовуємо average='macro'
acc = accuracy_score(y_test, y_pred_classes)
prec = precision_score(y_test, y_pred_classes, average='macro')
rec = recall_score(y_test, y_pred_classes, average='macro')
f1 = f1_score(y_test, y_pred_classes, average='macro')
cm = confusion_matrix(y_test, y_pred_classes)

print(f"Accuracy (Точність загальна): {acc:.4f}")
print(f"Precision (Точність за класами): {prec:.4f}")
print(f"Recall (Повнота): {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nConfusion Matrix (Матриця помилок):")
print(cm)
print("--------------------------------------")