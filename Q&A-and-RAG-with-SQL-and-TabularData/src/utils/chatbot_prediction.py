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

APPCFG = LoadConfig()


class ChatBotPrediction:
    """
    A ChatBot class capable of responding to messages using different modes of operation.
    It can interact with SQL databases, leverage language chain agents for Q&A,
    and use embeddings for Retrieval-Augmented Generation (RAG) with ChromaDB.
    """
    @staticmethod
    def respond(chatbot: List, message: str, chat_type: str, app_functionality: str, llm_model:str, llm_temperature:float, sql_mode:str) -> Tuple:
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
            if app_functionality == "Chat":
                print("In Chatbot Prediction")
                # If we want to use langchain agents for Q&A with our SQL DBs that was created from .sql files.
                if chat_type == "Q&A with stored SQL-DB":
                    # directories
                    if os.path.exists(APPCFG.sqldb_directory):
                        db = SQLDatabase.from_uri(
                            f"sqlite:///{APPCFG.sqldb_directory}")
                        execute_query = QuerySQLDataBaseTool(db=db)
                        write_query = create_sql_query_chain(
                            APPCFG.langchain_llm, db)
                        answer_prompt = PromptTemplate.from_template(
                            APPCFG.agent_llm_system_role)
                        answer = answer_prompt | APPCFG.langchain_llm | StrOutputParser()
                        chain = (
                            RunnablePassthrough.assign(query=write_query).assign(
                                result=itemgetter("query") | execute_query
                            )
                            | answer
                        )
                        response = chain.invoke({"question": message})

                    else:
                        chatbot.append(
                            (message, f"SQL DB does not exist. Please first create the 'sqldb.db'."))
                        return "", chatbot, None
                # If we want to use langchain agents for Q&A with our SQL DBs that were created from CSV/XLSX files.
                elif chat_type == "Q&A with Uploaded CSV/XLSX SQL-DB" or chat_type == "Q&A with stored CSV/XLSX SQL-DB":
                    if chat_type == "Q&A with Uploaded CSV/XLSX SQL-DB":
                        if os.path.exists(APPCFG.uploaded_files_sqldb_directory):
                            engine = create_engine(
                                f"sqlite:///{APPCFG.uploaded_files_sqldb_directory}")
                            db = SQLDatabase(engine=engine)
                            print(db.dialect)
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
                    
                    custom_prompt = """
                                    You are an agent designed to interact with a SQL database.
                                    Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
                                    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
                                    You can order the results by a relevant column to return the most interesting examples in the database.
                                    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
                                    You have access to tools for interacting with the database.
                                    Only use the given tools. Only use the information returned by the tools to construct your final answer.
                                    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

                                    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

                                    If the question does not seem related to the database, just return "I don't know" as the answer."""
                    
                    # prompt_template = PromptTemplate(input_variables=["query", "agent_scratchpad", "input", "dialect", "top_k"], template=custom_prompt)
                    
                    # agent_executor = create_sql_agent(
                    #     APPCFG.langchain_llm, db=db, agent_type="openai-tools", verbose=True, prompt=prompt_template)
                    # agent_executor = create_sql_agent(
                    #     APPCFG.langchain_llm, db=db, agent_type="openai-tools", verbose=True)
                    # response = agent_executor.invoke({"input": message})
                    # response = agent_executor.invoke({"input": custom_prompt.format(input=message)})
                    #-------------------------------------------------------------------
                    execute_query = QuerySQLDataBaseTool(db=db)
                    print('*'*30)
                    print("execute_query",execute_query)

                    ##Need to remove Comment
                    '''
                    write_query = create_sql_query_chain(
                            APPCFG.langchain_llm, db)
                    print('*'*30)
                    print("write_query",write_query)

                    answer_prompt = PromptTemplate.from_template(
                            APPCFG.agent_llm_system_role)
                    print('*'*30)
                    print("answer_prompt",answer_prompt)

                    answer = answer_prompt | APPCFG.langchain_llm | StrOutputParser()
                    print('*'*30)
                    print("answer",answer)

                    chain = (
                            RunnablePassthrough.assign(query=write_query).assign(
                                result=itemgetter("query") | execute_query
                            )
                            | answer
                        )
                    print('*'*30)
                    print("chain",chain)
                    '''
                    # --------------------------------------------------------------------
                    llm = APPCFG.langchain_llm

                    template = f"""You are an agent designed to interact with a SQL database.
                                    Given an input question, create a syntactically correct {db.dialect} query to run, then look at the results of the query and return the answer.
                                    Unless the user specifies a specific number of examples they wish to obtain, don't limit your query to at most {{top_k}} results.
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
                                You are an AI assistant skilled in data analysis and future prediction tasks. 
                                You will be given data based on a user's question and you are expected to:
                                1. Analyze the historical data .
                                2. Based on the insights, predict future outcomes or trends, using the most appropriate forecasting or prediction methods available to you.
                                3. Answer the user's question directly, providing actionable insights and predictions.
                                
                                Business Context:
                                The user will ask questions involving complex data analysis, often with a need for future predictions based on historical data.
                                Your goal is to generate extract the necessary data and then provide clear, actionable analysis and predictions based on this data.
                                Input Template:
                                SQL Query: {{query}}
                                User question: {{question}}
                                SQL Response: {{response}}

                                Please analyze the data and provide insights, including:
                                - if question is not related to forecast give sql answer straight a way
                                - More Accurate Predictions for the future based on historical data(Strictly, Don't give random text,symbol or blank/null as predicted/forecasted value)
                                - Actionable conclusions to answer the user's original question
                                - strictly If there is any tabular output display it as Table in proper format:.
                                - Give the forecasted output in tabular form.
                                - Strictly, Don't give random text,symbol or blank/null as predicted/forecasted value.
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
                    '''Use for normal query on database'''
                    ##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                    # write_query = create_sql_query_chain(
                    #         APPCFG.langchain_llm, db)
                    # answer_prompt = PromptTemplate.from_template(
                    #         APPCFG.agent_llm_system_role)
                    # answer = answer_prompt | APPCFG.langchain_llm | StrOutputParser()
                    # chain = (
                    #         RunnablePassthrough.assign(query=write_query).assign(
                    #             result=itemgetter("query") | execute_query
                    #         )
                    #         | answer
                    #     )
                    ##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

                    response = chain.invoke({"question": message})
                    response_1 = response

                    ''' Extract Python Code From Response.'''
                    # # response = response["output"]
                    # print("#"*100)
                    from IPython.display import display
                    display(response)
                    print("!"*100)
                    import re
                    import matplotlib.pyplot as plt
                    plt.switch_backend('TkAgg')
                    # code_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
                    # match = code_pattern.search(response)

                    # if match: 
                    #     extracted_code = match.group(1)
                    #     print("&"*100)
                    #     print(extracted_code) 
                    #     exec(extracted_code)
                    # else: 
                    #     print("Python code not found.")

                    ##################################################################################################################
                    from langchain.schema import HumanMessage, SystemMessage
                    messages = [
                                    SystemMessage(content="""You are an assistant tasked with generating Python code for visualizing given tabular data. Follow these instructions strictly:
                                        - Strictly use provided data — no external data should be considered.
                                        - Generate separate Python code for each table.
                                        - Don't consider the provided code — do not refer to or modify any prior code.
                                        - Only generate code for visualization — do not include any other functionality or logic.
                                        - Strictly generate 2 separate Python code snippets for the 2 tables — each visualization should be in a distinct block of code for each table.
                                        - Code format — the code should begin with ```python as usual.
                                        """),
                                    HumanMessage(content=f"{response}")
                                ]

                    response = (llm.invoke(messages)).content
                    print('@'*100)
                    print(response)
                    print('$'*200)
                    code_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
                    match = code_pattern.findall(response)

                    print('[]'*100)
                    try:
                        if match:
                            for idx, extracted_code in enumerate(match, 1):
                                print(f"Python code block {idx}:\n{extracted_code}\n")
                        
                                # extracted_code = match.group(1)
                                print("&"*100)
                                print(extracted_code) 
                                exec(extracted_code)
                        else: 
                            print("Python code not found.")
                    except Exception as e:
                        print(f"Error in python code execution:{e}")
                    
                    #####################################################################################################################

                    '''Pass response to chatgpt again and remove python code from it.'''   
                    from langchain.schema import HumanMessage, SystemMessage
                    messages = [
                                    SystemMessage(content="You are a helpful assistant.Strictly, Your task is to remove the python code from input context without changing anything.Refine the output in such a way that remove unnecessary output"),
                                    HumanMessage(content=f"{response_1}")
                                ]

                    response = llm.invoke(messages)
                    print('%'*200)
                    print(response.content)               
                    response= response.content

                    chatbot.append(
                    (message, response))
                    
                elif chat_type == "RAG with stored CSV/XLSX ChromaDB":
                    response = APPCFG.azure_openai_client.embeddings.create(
                        input=message,
                        model=APPCFG.embedding_model_name
                    )
                    query_embeddings = response.data[0].embedding
                    vectordb = APPCFG.chroma_client.get_collection(
                        name=APPCFG.collection_name)
                    results = vectordb.query(
                        query_embeddings=query_embeddings,
                        n_results=APPCFG.top_k
                    )
                    prompt = f"User's question: {message} \n\n Search results:\n {results}"

                    messages = [
                        {"role": "system", "content": str(
                            APPCFG.rag_llm_system_role
                        )},
                        {"role": "user", "content": prompt}
                    ]
                    llm_response = APPCFG.azure_openai_client.chat.completions.create(
                        model=APPCFG.model_name,
                        messages=messages
                    )
                    response = llm_response.choices[0].message.content

                # Get the `response` variable from any of the selected scenarios and pass it to the user.
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
            return "", chatbot
