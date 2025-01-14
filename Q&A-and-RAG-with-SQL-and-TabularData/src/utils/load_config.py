
import os
from dotenv import load_dotenv
import yaml
from pyprojroot import here
import shutil
from openai import AzureOpenAI
# from langchain.chat_models import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI
import chromadb
from langchain_community.llms import Ollama

print("Environment variables are loaded:", load_dotenv())


class LoadConfig:
    def __init__(self,llm_model="gpt-35-16k", llm_temperature = 0.7) -> None:
        with open(here("Q&A-and-RAG-with-SQL-and-TabularData/configs/app_config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        self.load_directories(app_config=app_config)
        self.load_llm_configs(app_config=app_config)
        self.load_openai_models(llm_model, llm_temperature)
        self.load_chroma_client()
        self.load_rag_config(app_config=app_config)

        # Un comment the code below if you want to clean up the upload csv SQL DB on every fresh run of the chatbot. (if it exists)
        # self.remove_directory(self.uploaded_files_sqldb_directory)

    def load_directories(self, app_config):
        self.stored_csv_xlsx_directory = here(
            app_config["directories"]["stored_csv_xlsx_directory"])
        self.sqldb_directory = str(here(
            app_config["directories"]["sqldb_directory"]))
        self.uploaded_files_sqldb_directory = str(here(
            app_config["directories"]["uploaded_files_sqldb_directory"]))
        self.stored_csv_xlsx_sqldb_directory = str(here(
            app_config["directories"]["stored_csv_xlsx_sqldb_directory"]))
        self.persist_directory = app_config["directories"]["persist_directory"]

    def load_llm_configs(self, app_config):
        # self.model_name = os.getenv("gpt_deployment_name")
        self.deployment_name = os.getenv("GET_DEPLOYMENT_NAME")
        self.model_name = os.getenv("MODEL_NAME")
        self.azure_endpoint=os.getenv("OPENAI_API_BASE"),
        self.api_key=os.getenv("OPENAI_API_KEY"),
        self.agent_llm_system_role = app_config["llm_config"]["agent_llm_system_role"]
        self.rag_llm_system_role = app_config["llm_config"]["rag_llm_system_role"]
        self.temperature = app_config["llm_config"]["temperature"]
        self.embedding_model_name = os.getenv("embed_deployment_name")

    def load_openai_models(self,llm_model, llm_temperature):
        if llm_model != 'llama3':
            azure_openai_api_key = os.environ["OPENAI_API_KEY"]
            azure_openai_endpoint = os.environ["OPENAI_API_BASE"]
            # This will be used for the GPT and embedding models
            self.azure_openai_client = AzureOpenAI(
                api_key=azure_openai_api_key,
                api_version=os.getenv("OPENAI_API_VERSION"),
                azure_endpoint=azure_openai_endpoint
            )

            self.langchain_llm = AzureChatOpenAI(
                openai_api_version=os.getenv("OPENAI_API_VERSION"),
                # azure_deployment=self.model_name,
                ## azure_deployment=self.deployment_name,
                ## model_name=self.model_name,
                azure_deployment = llm_model,
                model_name = llm_model,
                azure_endpoint=azure_openai_endpoint,
                api_key=azure_openai_api_key,
                temperature=llm_temperature
                # timeout = 60
                )
            self.client = AzureOpenAI(
                                    api_version=os.getenv("OPENAI_API_VERSION"),
                                    azure_endpoint=os.getenv("OPENAI_API_BASE"),
                                    api_key=os.getenv("OPENAI_API_KEY"),
                                    azure_deployment=os.getenv("GET_DEPLOYMENT_NAME")
                                )
        elif llm_model == 'llama3':
            self.langchain_llm = Ollama(model="llama3")
    

    def load_chroma_client(self):
        self.chroma_client = chromadb.PersistentClient(
            path=str(here(self.persist_directory)))

    def load_rag_config(self, app_config):
        self.collection_name = app_config["rag_config"]["collection_name"]
        self.top_k = app_config["rag_config"]["top_k"]

    def remove_directory(self, directory_path: str):
        """
        Removes the specified directory.

        Parameters:
            directory_path (str): The path of the directory to be removed.

        Raises:
            OSError: If an error occurs during the directory removal process.

        Returns:
            None
        """
        if os.path.exists(directory_path):
            try:
                shutil.rmtree(directory_path)
                print(
                    f"The directory '{directory_path}' has been successfully removed.")
            except OSError as e:
                print(f"Error: {e}")
        else:
            print(f"The directory '{directory_path}' does not exist.")
