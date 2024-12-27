import os
from typing import List, Tuple
from utils.load_config import LoadConfig
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
import langchain
langchain.debug = True

# APPCFG = LoadConfig()


class ChatBot:
    """
    A ChatBot class capable of responding to messages using different modes of operation.
    It can interact with SQL databases, leverage language chain agents for Q&A,
    and use embeddings for Retrieval-Augmented Generation (RAG) with ChromaDB.
    """
    @staticmethod
    def respond(chatbot: List, message: str, chat_type: str, app_functionality: str, llm_model:str, llm_temperature:float) -> Tuple:
        try:
            """
            Respond to a message based on the given chat and application functionality types.

            Args:
                chatbot (List): A list representing the chatbot's conversation history.
                message (str): The user's input message to the chatbot.
                chat_type (str): Describes the type of the chat (interaction with SQL DB or RAG).
                app_functionality (str): Identifies the functionality for which the chatbot is being used (e.g., 'Chat').

            Returns:
                Tuple[str, List, Optional[Any]]: A tuple containing an empty string, the updated chatbot conversation list,
                                                and an optional 'None' value. The empty string and 'None' are placeholder
                                                values to match the required return type and may be updated for further functionality.
                                                Currently, the function primarily updates the chatbot conversation list.
            """
            APPCFG = LoadConfig(llm_model, llm_temperature)
            if app_functionality == "Chat":
                # If we want to use langchain agents for Q&A with our SQL DBs that were created from CSV/XLSX files.
                if chat_type == "Q&A with stored SQL-DB" or chat_type == "Q&A with Uploaded CSV/XLSX SQL-DB" or chat_type == "Q&A with stored CSV/XLSX SQL-DB":
                    if chat_type == "Q&A with stored SQL-DB":
                        if os.path.exists(APPCFG.sqldb_directory):
                            engine = create_engine(
                                f"sqlite:///{APPCFG.sqldb_directory}")
                            db = SQLDatabase(engine=engine)
                        else:
                            chatbot.append(
                            (message, f"SQL DB does not exist. Please first create the 'sqldb.db'."))
                            return "", chatbot, None
                    elif chat_type == "Q&A with Uploaded CSV/XLSX SQL-DB":
                        if os.path.exists(APPCFG.uploaded_files_sqldb_directory):
                            engine = create_engine(
                                f"sqlite:///{APPCFG.uploaded_files_sqldb_directory}")
                            db = SQLDatabase(engine=engine)
                        else:
                            chatbot.append(
                                (message, f"SQL DB from the uploaded csv/xlsx files does not exist. Please first upload the csv files from the chatbot."))
                            return "", chatbot, None
                    elif chat_type == "Q&A with stored CSV/XLSX SQL-DB":
                        if os.path.exists(APPCFG.stored_csv_xlsx_sqldb_directory):
                            engine = create_engine(
                                f"sqlite:///{APPCFG.stored_csv_xlsx_sqldb_directory}")
                            db = SQLDatabase(engine=engine)
                        else:
                            chatbot.append(
                                (message, f"SQL DB from the stored csv/xlsx files does not exist. Please first execute `src/prepare_csv_xlsx_sqlitedb.py` module."))
                            return "", chatbot, None
                    
                    print(db.dialect)
                    print(db.get_usable_table_names())
                    
                    execute_query = QuerySQLDataBaseTool(db=db)
                    print('*'*30)
                    print("execute_query",execute_query)

                    llm = APPCFG.langchain_llm

                    template = f""" You are an agent designed to interact with a SQL database.
                                    Given an input question, create a syntactically correct {db.dialect} query to run, then look at the results of the query and return the answer.
                                    Unless the user specifies a specific number of examples they wish to obtain, strictly don't limit your query to at most {{top_k}} results.
                                    You can order the results by a relevant column to return the most interesting examples in the database.
                                    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
                                    You have access to tools for interacting with the database.
                                    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
                                    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

                                    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

                                    To start you should ALWAYS look at the tables in the database to see what you can query.
                                    Do NOT skip this step.
                                    Then you should query the schema of the most relevant tables.
                                    Here is the schema for the database:
                                        {{table_info}}
                                    Additional info: {{input}}
                                """

                    prompt = PromptTemplate.from_template(template)

                    sql_chain = create_sql_query_chain(llm, db, prompt)

                    template = f"""
                                You are an AI assistant skilled in answering user query.Whenever possible display data in tabular form.
                                Business Context:
                                Your goal is to generate, extract the necessary data and then provide clean and clear data.
                                Don't limit the data while answering the user query unless user asked. 
                                Input Template:
                                SQL Query: {{query}}
                                User question: {{question}}
                                SQL Response: {{response}}
                                """

                    prompt = PromptTemplate.from_template(template)
                    chain = (
                        RunnablePassthrough.assign(query=sql_chain).assign(
                            response=itemgetter("query") | QuerySQLDataBaseTool(db=db)
                        )
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                
                    response = chain.invoke({"question": message})

                chatbot.append(
                (message, response))

                return "", chatbot
            
            else:
                pass
        except Exception as e:
            error = str(e)
            print("Error:", error)
            chatbot.append(
                (message, f"Error:{error}"))
            return "", chatbot, None
        