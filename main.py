import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import utils
import random
import tkinter as tk
from PIL import ImageGrab


def create_data():  # функция для создания датасетов
    train_data = image_dataset_from_directory('dataset/train', seed=42, image_size=(30, 30), labels='inferred',
                                              label_mode='binary')
    test_data = image_dataset_from_directory('dataset/test', seed=42, image_size=(30, 30), labels='inferred',
                                             label_mode='binary')

    class_names = train_data.class_names  # создание классов
    x_train, y_train = zip(*train_data)
    x_train = np.concatenate(list(x_train), axis=0) / 255.0  # нормализация
    y_train = np.concatenate(list(y_train), axis=0)

    x_test, y_test = zip(*test_data)
    x_test = np.concatenate(list(x_test), axis=0) / 255.0  # нормализация
    y_test = np.concatenate(list(y_test), axis=0)
    return {'train': [x_train, y_train], 'test': [x_test, y_test], 'class_names': class_names}


def demonstrate_images(count, data):  # вывод count примеров изображений
    r1 = round(count ** (1 / 2))
    r2 = round(count ** (1 / 2))
    if r1 * r2 < count:
        r2 += 1
    for i in range(count):
        plt.subplot(r1, r2, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(data['train'][0][i], cmap=plt.cm.binary)
        plt.title(data['class_names'][int(data['train'][1][i])])
        plt.colorbar()
    plt.show()


def create_model(data):  # создание модели нейронной сети. data - словарь с данными
    CNT_NUERONS = 64  # количество нейронов в скрытом слое
    ACT = 'sigmoid'  # функция активации
    model = Sequential([
        Flatten(input_shape=(30, 30, 3)),  # преобразование многомерных данных в одномерный вектор
        Dense(CNT_NUERONS, activation=ACT),  # скрытый слой с CNT_NEURONS нейронами и функцией активации ACT
        Dense(len(data['class_names']), activation='softmax')
        # выходной слой с len(class_names) выходами и функцией активации softmax
        # (поскольку хочу получить наиболее вероятный класс)
    ])

    # приведение к вектору из 0 и 1 с размерностью len(class_names)
    y_train_cat = utils.to_categorical(data['train'][1], len(data['class_names']))
    y_test_cat = utils.to_categorical(data['test'][1], len(data['class_names']))
    data['train'].append(y_train_cat)
    data['test'].append(y_test_cat)

    # компиляция модели. поскольку 2 класса, функция потерь - 'binary_crossentropy'
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # обучение модели
    model.fit(data['train'][0], y_train_cat, epochs=20, validation_split=0.2, verbose=False)  # Обучение

    return model


def demonstrate_model(model):
    return model.summary()


def evoluate_model(data, model):  # оценка модели
    return model.evaluate(data['test'][0], data['test'][2], verbose=False)


def predict_el(data, model, img):  # предсказание для элемента img
    plt.imshow(img, cmap=plt.cm.binary)
    arr = np.expand_dims(img, axis=0)  # согласование размерности
    prediction = model.predict(arr)
    predicted_class = np.argmax(prediction)
    plt.title(data['class_names'][predicted_class])
    plt.show()
    return data['class_names'][predicted_class]


def error(data, model):  # выделение неверных результатов
    pred = np.argmax(model.predict(data['test'][0]), axis=1)
    mask = np.array([pred[i] == data['test'][1][i] for i in range(len(pred))])
    x_false = data['test'][0][mask[:, 0] == False]
    y_false = data['test'][1][mask[:, 0] == False]
    cnt = len(y_false)
    print('количество ошибок', cnt)
    if len(y_false) != 0:
        plt.title('Ошибочные результаты')
        r1 = round(cnt ** 1 / 2)
        r2 = round(cnt ** 1 / 2)
        if r1 * r2 < cnt:
            r2 += 1
        for i in range(cnt):
            plt.subplot(r1, r2, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x_false[i], cmap=plt.cm.binary)
            plt.title(data['class_names'][int(y_false[i])])
        plt.show()


def main():  # основная функция
    print('Здравствуйте')
    data = create_data()
    model = create_model(data)
    opts(data, model)


def opts(data, model):
    op = int(input('Что бы вы хотели посмотреть?:\n'
                   '1) Примеры изображений\n'
                   '2) Сводное представление модели\n'
                   '3) Оценка модели\n'
                   '4) Предсказание для одного элемента\n'
                   '5) Неверные результаты\n'
                   '6) Обработка Вашего рисунка(Прикладная задача)\n'
                   '7) Завершение программы\n'
                   'Введите номер операции: '))
    if op == 1:
        example(data, model)
    elif op == 2:
        demonstrate(data, model)
    elif op == 3:
        ev(data, model)
    elif op == 4:
        pred(data, model)
    elif op == 5:
        er(data, model)
    elif op == 6:
        paint(data, model)
    elif op == 7:
        print('Работа программы завершена')
    else:
        print('Неверный номер операции')


def example(data, model):
    count = int(input('Введите, сколько изображений Вы хотите видеть(от 1 до ' + str(len(data['train'][1])) + '): '))
    if count in range(1, len(data['train'][1] + 1)):
        demonstrate_images(count, data)
    else:
        print('Введено некорректное число')
    print()
    opts(data, model)


def demonstrate(data, model):
    print(demonstrate_model(model))
    print()
    opts(data, model)


def ev(data, model):
    loss, accuracy = evoluate_model(data, model)
    print('Потери(Test loss): ', loss)
    print('Точность(Test accuracy): ', accuracy)
    print()
    opts(data, model)


def pred(data, model):
    n = random.randint(0, len(data['test'][0]))  # номер изображения в наборе данных
    print("Предсказанный класс:" + predict_el(data, model, data['test'][0][n]))
    print()
    opts(data, model)


def er(data, model):
    error(data, model)
    print()
    opts(data, model)


def paint(data, model):  # функция для работы с изображением пользователя
    def click(event):  # функция для обработки процесса рисования
        x, y = event.x, event.y
        canvas.create_oval(x-10, y-10, x+10, y+10, fill='black')

    def save_image():  # сохранение изображения в numpy массив
        img = ImageGrab.grab(bbox=(
            canvas.winfo_rootx(), canvas.winfo_rooty(), canvas.winfo_rootx() + canvas.winfo_width(),
            canvas.winfo_rooty() + canvas.winfo_height()))
        img = img.resize((30, 30))  # приведение к 30x30 пикселям
        img = img.convert('RGB')
        img_array = np.array(img)
        np.save('image.npy', img_array)

    window = tk.Tk()  # создание окна
    window.title('Как Ваше настроение?')
    canvas = tk.Canvas(window, width=300, height=300, bg='white')
    canvas.pack()
    canvas.bind('<B1-Motion>', click)  # рисование
    button = tk.Button(window, text='Отправить', command=save_image)  # создание кнопки
    button.pack()
    window.mainloop()

    arr = np.load('image.npy') / 255.0  # выгрузка данных и нормализация
    pred_class = predict_el(data, model, arr)  # предсказание
    print(pred_class)
    # прикладная задача - вывод фраз
    if pred_class == 'smile':
        with open('smile_phrases.txt', 'r', encoding='UTF-8') as file:
            lines = [line.strip() for line in file]
            n = random.randint(0, len(lines)-1)
            print(lines[n])
    elif pred_class == 'sad':
        with open('sad_phrases.txt', 'r', encoding='UTF-8') as file:
            lines = [line.strip() for line in file]
            n = random.randint(0, len(lines)-1)
            print(lines[n])
    print()
    opts(data, model)


main()
