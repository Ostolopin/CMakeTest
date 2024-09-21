Репозиторий содержит исходные файла проекта и архив x64-Debug.zip с собранным проектом и  исполняемым файлом 3DObjectViewer

Описание:
Эта программа представляет собой 3D просмотрщик объектов, разработанный с использованием OpenGL, GLFW, GLEW и ImGui. Программа позволяет загружать файлы в формате .lprf, визуализировать 3D объекты на основе сечений, управлять их ориентацией и масштабом, а также создавать цилиндрическую модель на основе загруженного объекта. Для расчетов PCA используется библиотека Eigen, а для матричных операций - GLM.
Основные функции:

    Загрузка и отображение 3D моделей из файлов .lprf.
    Поворот, масштабирование и перемещение 3D модели.
    Создание цилиндрической аппроксимации модели.
    Просмотр объекта в режимах каркасной и сплошной визуализации.
    Интерактивный графический интерфейс (GUI) на базе ImGui для управления трансформациями и загрузки файлов.

Использование

    Загрузка модели: При запуске программы откроется диалоговое окно выбора файла. Выберите файл в формате .lprf для загрузки модели.
    Поворот/Масштаб/Перемещение модели:
        Поворот: Используйте ползунки в графическом интерфейсе или перетаскивайте мышь с зажатой левой кнопкой.
        Масштаб: Прокручивайте колесико мыши.
        Перемещение: Используйте клавиши "A" и "D" для перемещения модели вдоль оси X.
    Создание цилиндрической модели: Нажмите кнопку "Create a Cylinder" в графическом интерфейсе для создания цилиндрической аппроксимации загруженной модели.
    Режимы просмотра:
        По умолчанию объект отображается в каркасном режиме.
        Созданный цилиндр отображается в режиме сплошного заполнения.

Управление

    Перетаскивание мыши (Левая кнопка): Поворот модели.
    Колесико мыши: Масштабирование.
    Клавиши A/D: Перемещение модели вдоль оси X.
    Ползунки ImGui: Управление поворотом, масштабом и смещением.
