import os
import cv2
import numpy

# получаем путь к директории
dir_path = os.path.dirname(os.path.realpath(__file__))
# создаем папку Output, если она не существует
if not os.path.exists('Output'):
    os.makedirs('Output')
# импортируем модели, предоставленные в репозитории OpenCV
model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')
# просматриваем все файлы в папке
for file in os.listdir(dir_path):
    # разделяем имя файла и расширение на две части
    filename, file_extension = os.path.splitext(file)
    # проверяем, является ли расширение файла .png,.jpeg или .jpg, чтобы избежать чтения других файлов в каталоге
    if file_extension in ['.png', '.jpg', '.jpeg']:
        # чтение изображения с помощью cv2
        image = cv2.imread(file)
        # обращаемся к image.shape и берем первые два элемента, которые являются высотой и шириной
        (h, w) = image.shape[:2]
        # после вычитания среднего, нормализации и переключения каналов получаем блоб, который является нашим входным
        # изображением
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        # вводим блоб в модель и получаем обратное обнаружения со страницы с помощью model.forward()
        model.setInput(blob)
        detections = model.forward()
        # Итерация по всем обнаруженным лицам и извлечение их начальных и конечных точек
        count = 0
        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            confidence = detections[0, 0, i, 2]
            # если алгоритм более чем на 16,5% уверен, что обнаруженное лицо является лицом, то рисуем прямоугольник
            # вокруг него
            if confidence > 0.165:
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                count = count + 1
        # сохраняем измененное изображение в папке Output
        cv2.imwrite('Output/' + file, image)
        # выводим сообщение об успехе
        print("Face detection complete for image " + file + " (" + str(count) + ") faces found!")
