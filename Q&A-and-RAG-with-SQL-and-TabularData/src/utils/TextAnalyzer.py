import os, json
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
import instructor
from pydantic import BaseModel, Field
from enum import Enum
from typing import List
import pandas as pd
from IPython.display import display


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

langchain.debug = True

APPCFG = LoadConfig()


class TextAnalyzer:
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
                print("In Transcript Analyzer")
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
                    if os.path.exists(APPCFG.uploaded_files_sqldb_directory):
                        db_path = APPCFG.uploaded_files_sqldb_directory
                        engine = create_engine(
                            f"sqlite:///{APPCFG.uploaded_files_sqldb_directory}")
                        db = SQLDatabase(engine=engine)
                    else:
                        chatbot.append(
                            (message, f"SQL DB from the uploaded csv/xlsx files does not exist. Please first upload the csv files from the chatbot."))
                        return "", chatbot, None
                    
                    # chain = create_sql_query_chain(APPCFG.langchain_llm, db)
                    # response = chain.invoke({"question": message})
                    # print("Before:-",response)
                    # if llm_model == "llama3":
                    #     if 'sqlquery' in response.lower():
                    #         response = response.split("SQLQuery:", 1)[1].strip()
                    #     else:
                    #         chatbot.append(
                    #         (message, response))

                    #         return "", chatbot
                    # print("After:-",response)

                    import sqlite3  
                    connection = sqlite3.connect(db_path)  
                    cursor = connection.cursor()

                    # Execute the query
                    # cursor.execute(f"""SELECT transcript FROM Transcripts where callid = '{message}';""")
                    print("#"*50)
                    print(message)
                    print("#"*50)
                    cursor.execute(f"""SELECT transcript FROM Transcripts where callid='{message}';""")
                    columns = [description[0] for description in cursor.description]
                    result = cursor.fetchall()

                    # Now you have both columns and data
                    response = [dict(zip(columns, row)) for row in result]
                    print(response)
                    print(len(response))
                    print("$"*30)
                    cursor.close()
                    connection.close()

                    # Instructor makes it easy to get structured data like JSON from LLMs
                    client = instructor.patch(APPCFG.client)

                    # --------------------------------------------------------------
                    # Step 3: Define Pydantic data models
                    # --------------------------------------------------------------

                    """
                    This code defines a structured data model for classifying customer support tickets using Pydantic and Python's Enum class. 
                    It specifies categories, urgency levels, customer sentiments, and other relevant information as predefined options or constrained fields. 
                    This structure ensures data consistency, enables automatic validation, and facilitates easy integration with AI models and other parts of a support ticket system.
                    """

                    class CallCategory(str, Enum):
                        ORDER_ISSUE = "order_issue"
                        ACCOUNT_ACCESS = "account_access"
                        PRODUCT_INQUIRY = "product_inquiry"
                        TECHNICAL_SUPPORT = "technical_support"
                        BILLING = "billing"
                        DEVICEDAMAGE = "device_damage"
                        OTHER = "other"
                        

                    class CustomerSentiment(str, Enum):
                        ANGRY = "angry"
                        FRUSTRATED = "frustrated"
                        NEUTRAL = "neutral"
                        SATISFIED = "satisfied"

                    class AgentSentiment(str, Enum):
                        ANGRY = "angry"
                        FRUSTRATED = "frustrated"
                        NEUTRAL = "neutral"

                    class AgentBehavior(str, Enum):
                        HELPFUL = "helpful"
                        PATIENT = "patient"
                        IMPATIENT = "impatient"
                        AGGRESSIVE = "aggressive"
                        NEUTRAL = "neutral"
                        CONFIDENT = "confident"
                        APOLOGETIC = "apologetic"
                        PROFESSIONAL = "professional"

                    class AgentFaul(str, Enum):
                        FAUL = "faul"
                        NOFAUL = "no-faul"

                    class CustomerEsclation(str, Enum):
                        ESCALATION = "esclation"
                        NOESCALATION = "no-escalation"

                    class CallUrgency(str, Enum):
                        LOW = "low"
                        MEDIUM = "medium"
                        HIGH = "high"
                        CRITICAL = "critical"

                    class CallClassification(BaseModel):
                        category: CallCategory
                        urgency: CallUrgency
                        customer_sentiment: CustomerSentiment
                        agent_behavior:AgentBehavior
                        agent_faul : AgentFaul
                        customer_escalation : CustomerEsclation
                        agent_rating: float = Field(ge=0, le=5, description="Based on conversation give rating to agent")
                        confidence: float = Field(ge=0, le=1, description="Confidence score for the classification")
                        device_infromation: List[str] = Field(description="Based on conversation extract device information(like device details, model, device type etc.)")
                        key_information: List[str] = Field(description="Give 7-8 linear breif summary of the call")
                        suggested_action: str = Field(description="Brief suggestion for handling the ticket")

                    SYSTEM_PROMPT = """ You are an AI assistant for a claim management system at a Insurance Company.
                    Your role is to analyze incoming claim calls related to mobile devices and provide structured information to help our team respond quickly and effectively.

                    Business Context:

                    We handle thousands of claims daily, including warranty claims, damage claims, technical issues, and billing disputes.
                    Effective categorization, prioritization, and accurate responses are key to maintaining high customer satisfaction and operational efficiency.
                    Claims are processed based on urgency, warranty status, agent behaviour and customer sentiment.
                    Your Tasks:

                    Categorize the claim into the most appropriate category (e.g., device damage, warranty issue, technical issue, billing dispute).
                    Assess the urgency of the issue (low, medium, high, critical).
                    Determine the Customer's sentiment based on the tone of the message (positive, neutral, negative).
                    Determine the Agent's sentiment based on the tone of the message (positive, neutral, negative).
                    Determine the agent's behaviour based on the tone (helpful,patient).
                    Determine the agent's used faul language or not (faul or no-faul).
                    Determine the agent's rating between 1-5 (e.g 1-poor and 5-excellent).
                    Determine the wheather there is any escalation happend (escalation ,no-escalation).
                    7-8 line Breif Summary of the call that would be helpful for the claim processing team.
                    Based on conversation extract device information (like device details, model, device type etc.)
                    Suggest an initial action for processing the claim (e.g., inspect the device, escalate to technician, check warranty status).
                    Provide a confidence score for your classification based on the provided information.
                    Instructions:

                    Be objective and base your analysis solely on the information provided in the claim call conversation.
                    If unsure about any aspect, reflect that in your confidence score.
                    For key_information, extract specific details like device model, damage type, purchase date, serial number, or any additional facts.
                    The suggested_action should be clear and actionable to facilitate quick resolution.
                    Remember to consider the customer sentiment and adjust your urgency assessment accordingly.
                    """

                    def classify_ticket(ticket_text: str) -> CallClassification:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            response_model = CallClassification,
                            temperature=0,
                            max_retries=3,
                            messages=[
                                {
                                    "role": "system",
                                    "content": SYSTEM_PROMPT,
                                },
                                {"role": "user", "content": ticket_text}
                            ]
                        )
                        return response

                    if len(response) != 0:
                        result = classify_ticket("""{response[0]}""")
                        print(result.model_dump_json(indent=2))
                        df = pd.DataFrame([[response[0]]], columns=['conversation'])
                        df1 = pd.DataFrame([json.loads(result.model_dump_json(indent=2))])
                        df2 = pd.concat([df, df1], axis=1)
                        print("Display Dataframe")
                        display(df2)
                        headers = df2.columns.tolist()

                        # Create header row
                        header_row = "| " + " | ".join(headers) + " |"

                        # Create separator row
                        separator = "|-" + "-|-".join(["-" * len(header) for header in headers]) + "-|"

                        # Create rows for each record
                        rows = []
                        for _, record in df2.iterrows():
                            row = "| " + " | ".join(str(record[key]) for key in headers) + " |"
                            rows.append(row)

                        # Combine everything into a final string
                        table_string = header_row + "\n" + separator + "\n" + "\n".join(rows)

                        # Print or return the formatted table string
                        print(table_string)
                        response = table_string
                    else:
                        response = 'No data available.'

                chatbot.append(
                (message, response))

                return "", chatbot
        except Exception as e:
            error = str(e)
            print("Error:", error)
            chatbot.append(
                (message, f"Error:{error}"))
            return "", chatbot
