# TikTok Video Generator and Uploader.

## Overview
This project generates a video based on a user-provided text prompt, converts the text into speech, combines it with relevant images, and uploads the resulting video to TikTok.

-------------------------------

## Structure
The project consists of four main Python files:
1. **main.py**: The entry point of the application.
2. **merg_video.py**: Contains the `ImageAudioMerger` class for merging images and audio to create a video.
3. **video_generator.py**: Contains the `OpenaiConnector` class for interacting with the OpenAI API to generate content.
4. **tiktok_uploader.py**: Contains the `SendToTiktok` class for uploading the generated video to TikTok.

-------------------------------

## Requirements
- Python 3.7+
- moviepy
- PIL (Pillow)
- requests
- openai
- dotenv
- selenium
- selenium-stealth
- pyautogui
- aiohttp

-------------------------------

## Setup
1. Clone the repository to your local machine.
2. Create a virtual environment and activate it:
3. Install the required packages:
4. Create a `.env` file in the project root directory and add your OpenAI API key:

-------------------------------

## Usage
1. **Running the Application**:
- Run `main.py`:
  ```
  python main.py
  ```
- You will be prompted to enter a topic to generate a video about.

2. **main.py**:
- Initializes the main components and starts the process.
- Takes user input for the topic and generates the video.

3. **merg_video.py**:
- `ImageAudioMerger` class merges images and audio to create a video file.

4. **video_generator.py**:
- `OpenaiConnector` class handles interaction with the OpenAI API to generate text and images.
- Converts the generated text into speech.
- Combines the generated content into a video using `ImageAudioMerger`.

5. **tiktok_uploader.py**:
- `SendToTiktok` class automates the process of uploading the generated video to TikTok using Selenium and PyAutoGUI.

## Important Notes
- Ensure the paths and profile settings in `tiktok_uploader.py` are correctly set to match your environment.
- This project uses automation tools that may require specific configurations depending on your operating system and browser setup.

## Disclaimer
- This project is for educational purposes. Use it responsibly and ensure compliance with platform policies and guidelines.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.


Good luck