from models.llm import LargeLanguageModel
from groq import Groq
import os

class GroqModel(LargeLanguageModel):
    def prompt(self, 
               message, 
               *context):

        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": 
                        f'''## Question
                            {message}
                            ## Context
                            {context}''',
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