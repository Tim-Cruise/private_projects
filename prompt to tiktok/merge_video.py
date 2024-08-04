from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from pathlib import Path
from io import BytesIO
from PIL import Image
import requests
import os

class ImageAudioMerger:
    """
    Class to merge images and audio to create a video.
    """

    def __init__(self):
        pass

    def merge_and_create_video(self, speech, list_of_images, prompt_text):
        """
        Merge images and audio to create a video.

        Args:
            speech (str): Path to the speech audio file.
            list_of_images (list): List of image URLs.
            prompt_text (str): Text prompt used for generating the video title.

        Returns:
            str: Path to the created video file.
        """
        temp_image_paths = []
        audio = AudioFileClip(speech)
        total_duration = audio.duration
        duration_per_image = total_duration / len(list_of_images)

        clips = []
        for img_url in list_of_images:
            try:
                response = requests.get(img_url)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img = img.resize((1080, 1920), Image.LANCZOS)
                    img_path = f"images/temp_image_{list_of_images.index(img_url)}.png"
                    img.save(img_path)
                    temp_image_paths.append(img_path)
                    clips.append(ImageClip(img_path).set_duration(duration_per_image))
                else:
                    print(f"Failed to retrieve image from URL: {img_url}")
            except Exception as e:
                print(f"Error processing image {img_url}: {e}")

        if not clips:
            print("No valid images to create video.")
            return None

        video = concatenate_videoclips(clips, method="compose")
        video = video.set_audio(audio)
        output_path = f"video_to_post/10 fun facts about {prompt_text}.mp4"
        video.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=30)

        for img_path in temp_image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)

        if os.path.exists("speech.mp3"):
            os.remove("speech.mp3")

        return output_path
