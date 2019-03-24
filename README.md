# Darin

### План

I. Набросок архитектуры сети, создание интерфейса для отладки и взаимодействия, реализация поиска по дереву и работа с данными --- генерация выборки и обучение. Чтение статей об уже известных подходах для рендзю и игры "пять в ряд", реализация наиболее простых алгоритмов и выбор подходящих решений для будущей доработки нейросети. Сроки: 11 февраля -- 25 февраля.

II. Реализация начальной архитектуры сети. Обучение с учителем. Реализация интерфейса для турнира с другими сетями. Сроки: 25 февраля -- 11 марта.

III. Доработка сети с целью улучшения результата. Обучение с подкреплением и глубинное обучение для игры "пять в ряд". Сроки: 11 марта -- 21 марта.

### Алгоритм
Алгоритм построен на обученной сверточной нейронной сети, которая предсказывает ход профессионала исходя из расстановки на доске. Распределение нейронной сети используется в алгоритме Monte Carlo tree search.

### Игра
Для того, чтобы сыграть с алгоритмом, нужно скачать папку Game и запустить из нее скрипт gui_run.py. В нем следует написать за кого вы хотите играть: черные или белые. После этого откроется интерфейс. В нем следует нажимать на позиции на поле, куда игрок хочет сходить и ждать ход алгоритма около 3-4 секунд.

### Соревнование
Чтобы использовать алгоритм в соревновании, следует скачать папку Renju и использовать скрипты ментора проекта --- Симагина Дениса Андреевича, которые можно найти в его [репозитории](https://github.com/dasimagin/renju/tree/master/src). Достаточно скачать файл test.py, в нем изменить агента dummy.py на competitionAgzam.py и запустить скрипт.

### Технические требования
Потребууются следующие библиотеки:
* keras 2.2.4
* tensorflow >= 1.2
* python 3.5
* tkinter 4

