from dotenv import load_dotenv
import openai
import os

load_dotenv()

openai.api_key = os.getenv('CHATGPT_API_KEY')

def chatgpt_respons(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Replace with your engine of choice
        prompt=prompt,
        max_tokens=150  # Adjust as needed
    )
    
    respons_dict = response.get("choices")
    if respons_dict and len(respons_dict) > 0:
        prompt_respons = respons_dict[0]["text"]
    return prompt_respons