import logging
from models.llm import LargeLanguageModel
from groq import Groq
import os

class GroqModel(LargeLanguageModel):
    def __init__(self, logger: logging.Logger = logging.getLogger(__file__)):
        self.logger = logger
        
    def prompt(self, 
               message, 
               *context):

        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        self.logger.info(f"[GroqModel.prompt] prompt={message} prompt_ids={[doc.id for doc in context]}")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": 
                        f'''Answer the following question based on the context. 
                            Always and always (really important) add the context ID used as a reference to the information that helped to answer the underlying question.
                            Act normal as a search engine. Do not mention about context when you answer. 
                            If you can't answer, answer by the following template: "Unfortunately, no answer could be found for this prompt"
                            ## Question
                            {message}
                            ## Context
                            {[f"{doc.content}\n\
                                Context ID: {doc.id}" for doc in context]}''',
                }
            ],
            model="llama3-8b-8192",
        )
        
        return chat_completion.choices[0].message.content
    


if '__main__' == __name__:
    from dotenv import load_dotenv
    load_dotenv()
    
    model = GroqModel()
    print(model.prompt("", 'z'))