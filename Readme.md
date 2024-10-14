**Проект по двухэтапной сегментации баркодов**

Для запуска скриптов нужно скачать данные по адресу https://color.iitp.ru/index.php/s/xZyLFLFQoq2Rr4q и распаковать в корневую директорию репозетория.
Для запуска скриптов нужно воспользоваться версией Python 3.10.12 и установить пакеты из файла requirements.txt.

Для генерации вырезанного набора данных из исходных прямоугольников воспользуйтесь скриптом:
```
python cut_bbox.py -d "./data/yolo_barcode_dataset"
```
Скрипт генерирует для каждой поддиректории в соответствующей директории под-поддиректории с изображениями "\images_cut" и с разметкой "\markup_cut".

Для обучения детектора:
```
python train_yolo.py -c ./data/yolo_barcode_dataset/data.yaml -b yolo11m.pt
```

Для обучения сегментатора:
```
python train_yolo.py -c ./data/yolo_barcode_dataset/data.yaml -b yolo11m-seg.pt
```

Для обучения сегментатора на вырезанных баркодах:
```
python train_yolo.py -c ./data/yolo_barcode_dataset_cut/data.yaml -b yolo11m-seg.pt
```

Скрипт test_yolo.py подсчитывает среднее IoU для набора данных и считает метрики Precision, Recall, mAP50 и mAP50-95.
Отчёт 
Для получения метрик работы сегментационной модели на валидации:
```
python test_yolo.py -m ./data/models/segment_train/weights/best.pt -c ./data/yolo_barcode_dataset/data.yaml -d ./data/yolo_barcode_dataset/Validation
```

Для получения метрик работы сегментационной модели на тесте:
```
python test_yolo.py -m ./data/models/segment_train/weights/best.pt -c ./data/yolo_barcode_dataset/data_test.yaml -d ./data/yolo_barcode_dataset/Testing
```

Для получения метрик работы сегментационной модели для вырезанных баркодов на валидации:
```
python test_yolo.py -m ./data/models/segment_cut_train/weights/best.pt -c ./data/yolo_barcode_dataset_cut/data.yaml -d ./data/yolo_barcode_dataset_cut/Validation
```

Для получения метрик работы сегментационной модели для вырезанных баркодов на тесте:
```
python test_yolo.py -m ./data/models/segment_cut_train/weights/best.pt -c ./data/yolo_barcode_dataset_cut/data_test.yaml -d ./data/yolo_barcode_dataset_cut/Testing
```

Скрипт test_yolo_full_pipe.py запускает подряд детекционную модель, а потом на вырезанных детекциях запускает сегментатор. 
Скрипт подсчитывает среднее IoU для выбранного набора данных, а также сохраняет визуализацию предсказаний в указанную директорию.
Для визуализированных изображений зелёный прямоугольник -- результат детекции, синий многоугольник -- истинная граница сегментации, красная область -- результат работы сегментатора.
Для запуска на валидации: 
```
python test_yolo_full_pipe.py -md ./data/models/detection_train/weights/best.pt -ms ./data/models/segment_cut_train/weights/best.pt -d ./data/yolo_barcode_dataset/Validation -o ./output
```

Для запуска на тесте: 
```
python test_yolo_full_pipe.py -md ./data/models/detection_train/weights/best.pt -ms ./data/models/segment_cut_train/weights/best.pt -d ./data/yolo_barcode_dataset/Testing -o ./output
```