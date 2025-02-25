from utils.chatbot import ChatBot
from utils.chatbot_prediction import ChatBotPrediction
from utils.TextAnalyzer import TextAnalyzer
from typing import List, Tuple

''' This class will call another class based on sql_mode. '''
class ChatBotClass:
    @staticmethod
    def respond(chatbot: List, message: str, chat_type: str, app_functionality: str, llm_model:str, llm_temperature:float, sql_mode:str) -> Tuple:
        if sql_mode == 'Forecasting':
            return ChatBotPrediction.respond(chatbot, message, chat_type, app_functionality, llm_model, llm_temperature, sql_mode)
        elif sql_mode == 'Text Analyzer':
            return TextAnalyzer.respond(chatbot, message, chat_type, app_functionality, llm_model, llm_temperature, sql_mode)
        else:
            return ChatBot.respond(chatbot, message, chat_type, app_functionality, llm_model, llm_temperature, sql_mode)
