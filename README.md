# cv-epithelium

Анализ изображений эпителия методами компьютерного зрения

<img src="assets\gui.gif">

# How to run
1. Установить ```docker``` и ```docker-compose``` (на Win только ```docker```)
2. Запустить проект необходимо из корня командой ```docker build -f Dockerfile -t app:latest .```, параметр ```--build``` требуется вызвать только во время **самого первого** запуска
3. Выполнить команду ```docker run -p 8501:8501 app:latest```
4. Перейти на ```http://localhost:8501/``` в браузере
