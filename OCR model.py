"""
## Настройка
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import tensorflow as tf
import keras
from keras import ops
from keras import layers

# Путь к каталогу данных
data_dir = Path("./captcha_images/")

# Получить список всех изображений
images = sorted(list(map(str, list(data_dir.glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

print("Количество найденных изображений: ", len(images))
print("Количество найденных меток: ", len(labels))
print("Количество уникальных символов: ", len(characters))
print("Присутствующие символы: ", characters)

# Размер пакета для тренировки и валидации
batch_size = 16

# Желаемые размеры изображений
img_width = 200
img_height = 50

# Фактор, на который будет уменьшена исходная картинка
# блоками свертки. Мы будем использовать два
# блока свертки и каждый блок будет иметь
# слой подбора, который уменьшит функции в 2 раза.
# Следовательно, общий фактор понижения будет составлять 4.
downsample_factor = 4

# Максимальная длина любой капчи в наборе данных
max_length = max([len(label) for label in labels])


"""
## Предварительная обработка
"""
def binarize_image(image, threshold):
    # Конвертируем изображение в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Применяем пороговую обработку для получения бинарного изображения
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def remove_noise(image, kernel_size):
    # Применяем медианный фильтр для удаления шумов
    return cv2.medianBlur(image, kernel_size)

def blur_image(image, kernel_size):
    # Применяем размытие для сглаживания изображения
    return cv2.blur(image, (kernel_size, kernel_size))

def resize_image(image, width, height):
    # Изменяем размер изображения
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


# Сопоставление символов и целых чисел
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

# Сопоставление целых чисел с исходными символами
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def split_data(images, labels, train_size=0.9, shuffle=True):
    # 1. Получить общий размер датасета
    size = len(images)
    # 2. Создать индексный массив и перемешать его, если необходимо
    indices = ops.arange(size)
    if shuffle:
        keras.random.shuffle(indices)
    # 3. Получить размер обучающих образцов
    train_samples = int(size * train_size)
    # 4. Разделить данные на обучающие и валидационные наборы
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


# Разделение данных на обучающие и валидационные наборы
x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))


def encode_single_sample(img_path, label):
    # 1. Считать изображение
    img = tf.io.read_file(img_path)
    # 2. Декодировать и преобразовать в оттенки серого
    img = tf.io.decode_png(img, channels=1)
    # 3. Преобразовать в float32 в диапазоне [0, 1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Изменить размер до нужных размеров
    img = ops.image.resize(img, [img_height, img_width])
    # 5. Транспонировать изображение, потому что мы хотим, чтобы время
    # соответствовало ширине изображения.
    img = ops.transpose(img, axes=[1, 0, 2])
    # 6. Сопоставить символы в метке с числами
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 7. Вернуть словарь, так как наша модель ожидает два входа
    return {"image": img, "label": label}


"""
## Создание объектов `Dataset` 
"""


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


"""
## Визуализация данных
"""

_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch in train_dataset.take(1):
    images = batch["image"]
    labels = batch["label"]
    for i in range(16):
        img = (images[i] * 255).numpy().astype("uint8")
        label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
plt.show()

"""
## Модель
"""


def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    label_length = ops.cast(ops.squeeze(label_length, axis=-1), dtype="int32")
    input_length = ops.cast(ops.squeeze(input_length, axis=-1), dtype="int32")
    sparse_labels = ops.cast(
        ctc_label_dense_to_sparse(y_true, label_length), dtype="int32"
    )

    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())

    return ops.expand_dims(
        tf.compat.v1.nn.ctc_loss(
            inputs=y_pred, labels=sparse_labels, sequence_length=input_length
        ),
        1,
    )


def ctc_label_dense_to_sparse(labels, label_lengths):
    label_shape = ops.shape(labels)
    num_batches_tns = ops.stack([label_shape[0]])
    max_num_labels_tns = ops.stack([label_shape[1]])

    def range_less_than(old_input, current_input):
        return ops.expand_dims(ops.arange(ops.shape(old_input)[1]), 0) < tf.fill(
            max_num_labels_tns, current_input
        )

    init = ops.cast(tf.fill([1, label_shape[1]], 0), dtype="bool")
    dense_mask = tf.compat.v1.scan(
        range_less_than, label_lengths, initializer=init, parallel_iterations=1
    )
    dense_mask = dense_mask[:, 0, :]

    label_array = ops.reshape(
        ops.tile(ops.arange(0, label_shape[1]), num_batches_tns), label_shape
    )
    label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)

    batch_array = ops.transpose(
        ops.reshape(
            ops.tile(ops.arange(0, label_shape[0]), max_num_labels_tns),
            tf.reverse(label_shape, [0]),
        )
    )
    batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
    indices = ops.transpose(
        ops.reshape(ops.concatenate([batch_ind, label_ind], axis=0), [2, -1])
    )

    vals_sparse = tf.compat.v1.gather_nd(labels, indices)

    return tf.SparseTensor(
        ops.cast(indices, dtype="int64"),
        vals_sparse,
        ops.cast(label_shape, dtype="int64"),
    )


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        # Вычислить значение функции потерь в процессе обучения и добавить его
        # к слою с использованием `self.add_loss()`.
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(ops.shape(y_true)[1], dtype="int64")

        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # Во время тестирования просто возвращаем вычисленные прогнозы
        return y_pred


def build_model():
    # Входы в модель
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # Первый блок свертки
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Второй блок свертки
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Мы использовали два максимальных пула с размером пула и шагом 2.
    # Следовательно, субдискретизированные карты признаков уменьшаются в 4 раза. Количество фильтров
    # в последнем слое - 64. Соответствующим образом изменяем форму перед
    # передача результатов в часть RNN модели
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # Выходной слой
    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

    # Добавляем слой CTC для вычисления потери CTC на каждом шаге
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Определение модели
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Оптимизатор
    opt = keras.optimizers.Adam()
    # Компилирование модели и возвращение
    model.compile(optimizer=opt)
    return model


# Получить модель
model = build_model()
model.summary()

"""
## Обучение
"""


epochs = 100
early_stopping_patience = 10
# Добавить раннюю остановку
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Обучение модели
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping],
)

def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    input_shape = ops.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())
    input_length = ops.cast(input_length, dtype="int32")

    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length
        )
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    return (decoded_dense, log_prob)


# Получить модель прогнозирования, извлекая слои до выходного слоя
prediction_model = keras.models.Model(
    model.input[0], model.get_layer(name="dense2").output
)
prediction_model.summary()


# Вспомогательная функция для декодирования вывода сети
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Используем жадный поиск. Для сложных задач можно использовать поиск луча
    results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Итерация по результатам и возврат текста
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


#  Проверим результаты на некоторых примерах валидации
for batch in validation_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(len(pred_texts)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f"Прогноз: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")

plt.show()
