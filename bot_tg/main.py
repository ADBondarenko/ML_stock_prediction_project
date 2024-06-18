import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command, CommandObject
from pydantic import BaseModel
import os
import requests
import json
import logging

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token=os.getenv('BOT_TOKEN'))
API_URL = os.getenv('API_URL_SERVICE')
API_URL = f"http://"+API_URL
# Диспетчер
dp = Dispatcher()
# # @ml_app.get('/')
# # @ml_app.get('/model/get_model_by_name')
# # @ml_app.get('/model/get_new_model')
# # @ml_app.get('/model/get_all_availible_models')
# # @ml_app.get('/model/get_models_by_ticker')
# @ml_app.get('/model/predict_by_model_ticker')
# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

# Хэндлер на команду /start
@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Hello, financial-modelling enjoyer! \n"
                         "Here is the list of availible commands up to date: \n"
                        "/get_service_status -- checks whether the service is running \n"
                        "/get_model_by_name -- <name> get .pkl of that specific model \n"
                        "/get_new_model -- get a new model by <ticker>, <timeframe>, <model_type>, <num_bars_back> \n"
                        "Опциональные аргументы для генерации фичей (порядок НУЖНО соблюдать) \n"
                         "binary_rsi : bool = True, rsi_period : int =14, \n"
                         "rsi_levels : list = [20,40,60,80], binary_ema : bool = True\n"
                         "ema_periods : list = [8,24], nth_diff : int = 1 \n"
                        "/get_models_by_ticker <ticker> \n"
                        "/get_all_model -- go a little window-shopping for availible models\n"
                        "remember to set <most_recent> parameter to False for a full-list \n"
                        "/predict_by_model_ticker -- get a prediction for given <model_name>, <ticker>,"
                         "<timeframe> and <num_bars_back> \n"
                        "\n"
                        "Please remember to provide params in a whitespace-separated 'param_1 param_2' fashion")
    
@dp.message(Command("get_service_status"))
async def get_service(message: types.Message):
    base_url = API_URL
    method_url = "/"
    #No payload for this request.
    
    request_url = base_url+method_url
    reply = requests.get(request_url).content

    reply_text = f"Ответ сервиса: {json.loads(reply)}"

    await message.reply(reply_text)

@dp.message(Command("get_model_by_name"))
async def get_model_by_name(message: types.Message,
                           command : CommandObject):
    if command.args is None:
                await message.answer(
                    "Ошибка: аргументы не предоставлены. Правильно:\n"
                    "/get_new_model <name>\n"
                    "Актуальный список моделей можно получить по команде /get_all_model"
                )
                return      
    else:
        try:
            args_list = command.args.split(" ", maxsplit=-1)
            if len(args_list) != 1:
                raise ValueError
        # Если получилось меньше двух частей, вылетит ValueError
        except ValueError:
            await message.answer(
                "Ошибка: аргументы не предоставлены. Правильно:\n"
                "/get_new_model <name>\n"
                "Актуальный список моделей можно получить по команде /get_all_model"
            )
            return
        name = args_list[0]
        base_url = API_URL
        method_url = f"/model/get_model_by_name?name={name}"
        request_url = base_url+method_url
        
        reply = requests.get(request_url+method_url).content
    
        reply_text = f"Ответ сервиса: {json.loads(reply)}"

    await message.reply(reply_text)

@dp.message(Command("get_new_model"))
async def get_new_model(message: types.Message,
                   command : CommandObject):

    #No payload for this request.
    if command.args is None:
            await message.answer(
                "Ошибка: аргументы не предоставлены. Правильно:\n"
                "/get_new_model <ticker>, <timeframe>, <model_type>, <num_bars_back> \n"
                "<model_type> бывает     RandomForest  - 'rf', \n"
                "LinearRegression = 'lr' \n"
                "HistGradientBoosting = 'hgb' \n"
                "MultiLevelPerceptron = 'mlp' \n"
                "Актуальный список тикеров лежит на сайте OKX/в утилях"
            )
            return      
    else:
        try:
            args_list = command.args.split(" ", maxsplit=-1)
            if len(args_list) != 4:
                raise ValueError
        # Если получилось меньше двух частей, вылетит ValueError
        except ValueError:
            await message.answer(
                "Ошибка: аргументы не предоставлены. Правильно:\n"
                "/get_new_model <ticker>, <timeframe>, <model_type>, <num_bars_back> \n"
                "<model_type> бывает     RandomForest  - 'rf', \n"
                "LinearRegression = 'lr' \n"
                "HistGradientBoosting = 'hgb' \n"
                "MultiLevelPerceptron = 'mlp' \n"
                "LSTM = 'lstm' \n"
                "Актуальный список тикеров лежит на сайте OKX/в утилях"
            )
            return
        ticker = args_list[0]
        timeframe = args_list[1]
        model_type = args_list[2]
        num_bars_back = args_list[3]
        
        base_url = API_URL
        method_url = f"/model/get_new_model/?ticker={ticker}&timeframe={timeframe}&model_type={model_type}&num_bars_back={num_bars_back}"
        request_url = base_url+method_url
        
        reply = requests.get(request_url+method_url).content
    
        reply_text = f"Ответ сервиса: {json.loads(reply)}"

    await message.reply(reply_text)

@dp.message(Command("get_all_model"))
async def get_all_model(message: types.Message,
                   command : CommandObject):

    #No payload for this request.
    base_url = API_URL
    if command.args is None:
        most_recent = True
        await message.answer(
            "Используем значение по умолчанию - <most_recent> == True")
    else:
        try:
            args_list = command.args.split(" ", maxsplit=-1)
            if len(args_list) != 1:
                raise ValueError
            most_recent = args_list[0]

        # Если получилось меньше двух частей, вылетит ValueError
        except ValueError:
            await message.answer(
                "Ошибка: аргументы не предоставлены. Правильно:\n"
                "/get_new_model <most_recent> : True/False \n"
            )
            return
    
    method_url = f"/model/get_all_availible_models?most_recent={most_recent}"
    request_url = base_url+method_url
    
    reply = requests.get(request_url+method_url).content

    reply_text = f"Ответ сервиса: {json.loads(reply)}"

    await message.reply(reply_text)
    
@dp.message(Command("get_models_by_ticker"))
async def get_models_by_ticker(message: types.Message,
                   command : CommandObject):

    if command.args is None:
        await message.answer(
                "Ошибка: аргументы не предоставлены. Правильно:\n"
                "/get_models_by_ticker <ticker> /n"
        )
        return
    try:
        args_list = command.args.split(" ", maxsplit = -1)
        if len(args_list) != 1:
            raise ValueError
        ticker = args_list[0]

    except ValueError:
        await message.answer(
            "Ошибка: неправильный формат команды. Правильно:\n"
            "/get_models_by_ticker <ticker> \n"
        )
        return
    base_url = API_URL
    method_url = f"/model/get_models_by_ticker/?ticker={ticker}"
    #Hardcoded payload
    
    request_url = base_url+method_url
    reply = requests.get(request_url).content

    reply_text = f"Ответ сервиса: {json.loads(reply)}"

    await message.reply(reply_text)
#requests.patch(url, params={key: value}, args)

    await message.reply(reply_text)


@dp.message(Command("predict_by_model_ticker"))
async def predict_by_model_ticker(message: types.Message,
                   command : CommandObject):

    #No payload for this request.
    if command.args is None:
            await message.answer(
                "Ошибка: аргументы не предоставлены. Правильно:\n"
                "/predict_by_model_ticker <model_name>, <ticker>, <timeframe>, <num_bars_back>  \n"
                "<num_bars_back> \n"
            )
            return      
    else:
        try:
            args_list = command.args.split(" ", maxsplit=-1)
            if (len(args_list) != 3) or (len(args_list) != 4):
                raise ValueError
            if len(args_list) == 3:
                model_name = args_list[0]
                ticker = args_list[1]
                timeframe = args_list[2]
            if len(args_list) == 4:
                model_name = args_list[0]
                ticker = args_list[1]
                timeframe = args_list[2]
                num_bars_back = args_list[3]
        # Если получилось меньше двух частей, вылетит ValueError
        except ValueError:
            await message.answer(
                "Ошибка: аргументы не предоставлены. Правильно:\n"
                "/predict_by_model_ticker <ticker>, <timeframe>, <model_type>, <num_bars_back>  \n"
                "<num_bars_back> \n"
            )
            return

        base_url = API_URL
        method_url = f"/model/predict_by_model_ticker/?ticker={ticker}&timeframe={timeframe}&model_name={model_type}&num_bars_back={num_bars_back}"
        request_url = base_url+method_url
        
        reply = requests.get(request_url+method_url).content
    
        reply_text = f"Ответ сервиса: {json.loads(reply)}"

    await message.reply(reply_text)


if __name__ == "__main__":
    asyncio.run(main())




