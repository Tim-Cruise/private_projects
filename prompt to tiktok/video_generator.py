import os
import openai
from dotenv import load_dotenv
import asyncio
import requests
import aiohttp
from pathlib import Path
import re
from merg_video import ImageAudioMerger

class OpenaiConnector:
    """
    Class to interact with OpenAI API and generate video content.
    """

    def __init__(self):
        """
        Initialize the OpenaiConnector class.
        """
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.merger = ImageAudioMerger()
        self.assistant_ids = {
            "text": "asst_QtnZtWlejRmLeoGeO0ctdZkn",
            "image": "asst_zgLJeAskvkE96CFVTGzelmef",
            "optimize text": "asst_65iajv4N52lHbHixu8Zl25sV",
            "optimize image": "asst_zgLJeAskvkE96CFVTGzelmef",
            "hashtag": "asst_dbB8Tw0YHqC3dFsvhJhJ7gAP"
        }

    async def prompt_bot(self, instructions, assistant_type):
        """
        Prompt the OpenAI bot with instructions.

        Args:
            instructions (str): Instructions for the bot.
            assistant_type (str): Type of assistant to use.

        Returns:
            str: Response from the assistant.
        """
        thread = self.client.beta.threads.create()
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.assistant_ids[assistant_type],
            instructions=instructions
        )
        
        while True:
            run_status = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_status.status == "completed":
                break
            await asyncio.sleep(1)

        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        api_responses = [msg for msg in messages.data if msg.role == "assistant"]

        if api_responses:
            return api_responses[-1].content
        else:
            return "No response from assistant."

    def generate_images(self, list_of_facts):
        """
        Generate images using OpenAI.

        Args:
            list_of_facts (list): List of facts to generate images for.

        Returns:
            list: List of URLs to generated images.
        """
        list_of_images = []
    
        for fact in list_of_facts:
            try:
                response = self.client.images.generate(
                    model="dall-e-3",
                    prompt=fact,
                    n=1, 
                    size="1024x1024",
                    quality="standard"
                )
                list_of_images.append(response.data[0].url)
            except Exception as e:
                return f"Error in generating images: {str(e)}"
            
        return list_of_images

    def text_to_speech(self, text, model="tts-1-hd"):
        """
        Convert text to speech using OpenAI.

        Args:
            text (str): Text to convert to speech.
            model (str): Model to use for text-to-speech conversion.

        Returns:
            str: Path to the generated speech audio file.
        """
        try:
            speech_file_path = Path(__file__).parent / "speech.mp3"
            response = self.client.audio.speech.create(
                model=model,
                voice="alloy",
                input=text
            )
            response.stream_to_file(speech_file_path)
            return str(speech_file_path)
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def split_text_into_segments(self, text):
        """
        Split text into segments based on numbered points.

        Args:
            text (str): Text to split.

        Returns:
            list: List of text segments.
        """
        pattern = re.compile(r'(\d+\..*?)(?=\n\n|\Z)', re.DOTALL)
        matches = pattern.findall(text)
        segments = [match.strip() for match in matches]
        return segments

    def remove_before_one(self, text):
        """
        Remove text before the first numbered point.

        Args:
            text (str): Text to process.

        Returns:
            str: Processed text.
        """
        match = re.search(r'\b1\b', text)
        if match:
            return text[match.start():]
        return text

    def run(self, prompt_text):
        """
        Run the process of generating content using OpenAI.

        Args:
            prompt_text (str): Text prompt for generating content.
        """
        prompt = f"I want 10 facts about {prompt_text}"
        print("get text from AI")
        ai_text_respond = asyncio.run(self.prompt_bot(prompt, "optimize text"))
        ai_text_respond = ai_text_respond[0].text.value
        ai_text_respond_list = self.split_text_into_segments(ai_text_respond)

        print("Generate pictures")
        ai_images_respond_list = self.generate_images(ai_text_respond_list)

        text_for_speech = self.remove_before_one(ai_text_respond)
        
        print("Create speech file")
        ai_speech = self.text_to_speech(text_for_speech)

        print("Make mp4")
        compleat_video = self.merger.merge_and_create_video(ai_speech, ai_images_respond_list, prompt_text)

if __name__ == "__main__":
    openai_connector = OpenaiConnector()
    print("START")
    openai_connector.run("")
    print("END")
