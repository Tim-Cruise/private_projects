from merg_video import ImageAudioMerger
from tiktok_uploader import SendToTiktok
from video_generator import OpenaiConnector

class PromptToTiktok:
    """
    Main class to generate a video from a text prompt and upload it to TikTok.
    """

    def __init__(self) -> None:
        """
        Initialize the PromptToTiktok class.
        """
        self.merger = ImageAudioMerger()
        self.uploader = SendToTiktok()
        self.openai_connector = OpenaiConnector()

    def run(self):
        """
        Run the process of generating and uploading a video.
        """
        print("START")
        
        # Get the text prompt from the user
        text_prompt = input("What do you want to learn 10 things about? ")
        
        # Generate the video
        self.openai_connector.run(text_prompt)
        
        # Upload the video to TikTok
        self.uploader.run(text_prompt)

        print("END")

if __name__ == "__main__":
    ptt = PromptToTiktok()
    ptt.run()
