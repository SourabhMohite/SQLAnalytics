import gradio as gr
from utils.upload_file import UploadFile
from utils.chatbot import ChatBot
from utils.chatbot_prediction import ChatBotPrediction
from utils.chatbotclass import ChatBotClass
from utils.ui_settings import UISettings

theme = gr.themes.Ocean(primary_hue="indigo", secondary_hue="green", neutral_hue="slate",).set(
    block_background_fill='*color_accent_soft',
    block_background_fill_dark='*button_secondary_background_fill_hover',
    block_border_color='*checkbox_background_color_selected',
    block_info_text_color='*slider_color',
    block_label_background_fill='*input_background_fill',
    block_label_background_fill_dark='*stat_background_fill',
    block_label_border_color='*block_title_border_color',
    block_label_text_color='*checkbox_border_color_focus',
    block_title_background_fill_dark='*button_secondary_background_fill',
    block_title_border_width_dark='*input_border_width'
)
########################################################################################################################################### 
'''Q&A on SQL Database'''
########################################################################################################################################### 
with gr.Blocks(theme=theme) as demo:
#with gr.Blocks(gr.themes.Ocean(primary_hue="indigo", secondary_hue="green", neutral_hue="slate",)) as demo:
    with gr.Tabs():
        with gr.TabItem("Asurion SQL Analytics CHATBOT"):
            gr.Markdown("<h1 style='color: #FF69B4;'>Asurion SQL Analytics CHATBOT</h1>")
            with gr.Row():
                # Left column for the radio button
                with gr.Column(scale=2, min_width=350, elem_id="left_column"):
                    llm_model = gr.Radio(
                        choices=["llama3","gpt-35-16k", "gpt-4"], 
                        value = "llama3",
                        label="LLM Models", 
                        info="Choose your LLM Model", 
                        interactive=True
                    )
                    llm_temperature = gr.Slider(0, 1, value=0.7, label="Temperature", interactive=True, info="Choose the temperature")

                    app_functionality = gr.Dropdown(
                    label="App functionality", choices=["Chat", "Process files"], value="Chat",interactive=True)

                    chat_type = gr.Dropdown(
                    label="Chat type", choices=[
                        "Q&A with stored SQL-DB",
                        # "Q&A with stored CSV/XLSX SQL-DB",
                        "Q&A with Uploaded CSV/XLSX SQL-DB",
                        "Acyan Redshift"
                    ], value="Q&A with stored SQL-DB",interactive=True)

                    upload_btn = gr.UploadButton(
                        "📁 Upload CSV or XLSX files", file_types=['.csv'], file_count="multiple")
                    
                    sql_mode = gr.Radio(
                        choices=["SQL Only", "SQL Analytics", "Forecasting", "Text Analyzer"], 
                        value = "SQL Only",
                        label="Mode", 
                        info="Choose Your Query Mode", 
                        interactive=True
                    )

                    # database_conn = gr.Textbox(
                    # lines=1,
                    # scale=2,
                    # placeholder="Enter Database Connection Here",
                    # container=False,
                    # )
                
                # Right column for the main content
                with gr.Column(scale=8, min_width=900, elem_id="right_column"):
                    ##############
                    # First ROW:
                    ##############
                    with gr.Row() as row_one:
                        chatbot = gr.Chatbot(
                            [],
                            elem_id="chatbot",
                            bubble_full_width=False,
                            height=500,
                            avatar_images=(
                                ("Q&A-and-RAG-with-SQL-and-TabularData/images/AI_RT.png"), "Q&A-and-RAG-with-SQL-and-TabularData/images/openai.png")
                        )
                        # **Adding like/dislike icons
                        chatbot.like(UISettings.feedback, None, None)
                
            ##############
            # SECOND ROW:
            ##############
            with gr.Row():
                input_txt = gr.Textbox(
                    lines=4,
                    scale=8,
                    placeholder="Enter text and press enter, or upload PDF files",
                    container=False,
                )
            ##############
            # Third ROW:
            ##############
            with gr.Row() as row_two:
                text_submit_btn = gr.Button(value="Submit text")
                # upload_btn = gr.UploadButton(
                #     "📁 Upload CSV or XLSX files", file_types=['.csv'], file_count="multiple")
                # app_functionality = gr.Dropdown(
                #     label="App functionality", choices=["Chat", "Process files"], value="Chat")
                # chat_type = gr.Dropdown(
                #     label="Chat type", choices=[
                #         "Q&A with stored SQL-DB",
                #         "Q&A with stored CSV/XLSX SQL-DB",
                #         "Q&A with Uploaded CSV/XLSX SQL-DB"
                #     ], value="Q&A with stored SQL-DB")
                clear_button = gr.ClearButton([input_txt, chatbot])
            ##############
            # Process:
            ##############
            file_msg = upload_btn.upload(fn=UploadFile.run_pipeline, inputs=[
                upload_btn, chatbot, app_functionality], outputs=[input_txt, chatbot], queue=False)
            
            

            txt_msg = input_txt.submit(fn=ChatBotClass.respond,
                                       inputs=[chatbot, input_txt,
                                               chat_type, app_functionality, llm_model, llm_temperature, sql_mode],
                                       outputs=[input_txt,
                                                chatbot],
                                       queue=False).then(lambda: gr.Textbox(interactive=True),
                                                         None, [input_txt], queue=False)

            txt_msg = text_submit_btn.click(fn=ChatBotClass.respond,
                                            inputs=[chatbot, input_txt,
                                                    chat_type, app_functionality, llm_model, llm_temperature, sql_mode],
                                            outputs=[input_txt,
                                                     chatbot],
                                            queue=False).then(lambda: gr.Textbox(interactive=True),
                                                              None, [input_txt], queue=False)
    """     
    ###########################################################################################################################################        
    ''' Forecating Engine '''
    ########################################################################################################################################### 
    
    with gr.Tabs():
        with gr.TabItem("Forecasting Engine"):
            gr.Markdown("<h1 style='color: #FF69B4;'>Asurion Forecasting Engine</h1>")
            with gr.Row():
                # Left column for the radio button
                with gr.Column(scale=2, min_width=150, elem_id="left_column"):
                    llm_model = gr.Radio(
                        choices=["gpt-35-16k", "gpt-4"], 
                        value = "gpt-35-16k",
                        label="LLM Models", 
                        info="Choose your LLM Model", 
                        interactive=True
                    )
                    llm_temperature = gr.Slider(0, 1, value=0.7, label="Temperature", interactive=True, info="Choose the temperature")

                    app_functionality = gr.Dropdown(
                    label="App functionality", choices=["Chat", "Process files"], value="Chat",interactive=True)

                    chat_type = gr.Dropdown(
                    label="Chat type", choices=[
                        "Q&A with stored SQL-DB",
                        "Q&A with stored CSV/XLSX SQL-DB",
                        "Q&A with Uploaded CSV/XLSX SQL-DB"
                    ], value="Q&A with stored SQL-DB",interactive=True)

                    upload_btn = gr.UploadButton(
                        "📁 Upload CSV or XLSX files", file_types=['.csv'], file_count="multiple")
                    
                    visual_graph = gr.Radio(
                        choices=["Bar Chart", "Line Chart"], 
                        value = "Bar Chart",
                        label="Graphs", 
                        info="Choose your visual", 
                        interactive=False
                    )

                    # database_conn = gr.Textbox(
                    # lines=1,
                    # scale=2,
                    # placeholder="Enter Database Connection Here",
                    # container=False,
                    # )
                
                # Right column for the main content
                with gr.Column(scale=8, min_width=500, elem_id="right_column"):
                    ##############
                    # First ROW:
                    ##############
                    with gr.Row() as row_one:
                        chatbot = gr.Chatbot(
                            [],
                            elem_id="chatbot",
                            bubble_full_width=False,
                            height=500,
                            avatar_images=(
                                ("Q&A-and-RAG-with-SQL-and-TabularData/images/AI_RT.png"), "Q&A-and-RAG-with-SQL-and-TabularData/images/openai.png")
                        )
                        # **Adding like/dislike icons
                        chatbot.like(UISettings.feedback, None, None)
                
            ##############
            # SECOND ROW:
            ##############
            with gr.Row():
                input_txt = gr.Textbox(
                    lines=4,
                    scale=8,
                    placeholder="Enter text and press enter, or upload PDF files",
                    container=False,
                )
            ##############
            # Third ROW:
            ##############
            with gr.Row() as row_two:
                text_submit_btn = gr.Button(value="Submit text")
                # upload_btn = gr.UploadButton(
                #     "📁 Upload CSV or XLSX files", file_types=['.csv'], file_count="multiple")
                # app_functionality = gr.Dropdown(
                #     label="App functionality", choices=["Chat", "Process files"], value="Chat")
                # chat_type = gr.Dropdown(
                #     label="Chat type", choices=[
                #         "Q&A with stored SQL-DB",
                #         "Q&A with stored CSV/XLSX SQL-DB",
                #         "Q&A with Uploaded CSV/XLSX SQL-DB"
                #     ], value="Q&A with stored SQL-DB")
                clear_button = gr.ClearButton([input_txt, chatbot])
            ##############
            # Process:
            ##############
            file_msg = upload_btn.upload(fn=UploadFile.run_pipeline, inputs=[
                upload_btn, chatbot, app_functionality], outputs=[input_txt, chatbot], queue=False)

            txt_msg = input_txt.submit(fn=ChatBotPrediction.respond,
                                       inputs=[chatbot, input_txt,
                                               chat_type, app_functionality],
                                       outputs=[input_txt,
                                                chatbot],
                                       queue=False).then(lambda: gr.Textbox(interactive=True),
                                                         None, [input_txt], queue=False)

            txt_msg = text_submit_btn.click(fn=ChatBotPrediction.respond,
                                            inputs=[chatbot, input_txt,
                                                    chat_type, app_functionality],
                                            outputs=[input_txt,
                                                     chatbot],
                                            queue=False).then(lambda: gr.Textbox(interactive=True),
                                                              None, [input_txt], queue=False)

    """
if __name__ == "__main__":
    demo.launch()
