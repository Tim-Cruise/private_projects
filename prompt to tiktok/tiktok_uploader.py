from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
import time
import pyautogui
import os
from selenium_stealth import stealth
from random import uniform

class SendToTiktok:
    """
    Class to upload videos to TikTok using Selenium and PyAutoGUI.
    """

    def __init__(self) -> None:
        """
        Initialize the SendToTiktok class.
        """
        self.path_to_mp4 = "path/to/your/mp4/directory" # Update this path accordingly
        self.video_name = "tiktok_video.mp4" # The video name
        self.driver = None

    def random_time(self):
        """
        Sleep for a random time between 0 to 3 seconds.
        """
        t = round(uniform(0, 3), 2)
        time.sleep(t)

    def load_mp4(self, prompt_text):
        """
        Load the MP4 file using PyAutoGUI.

        Args:
            prompt_text (str): Text prompt to generate the video file name.
        """
        self.random_time()  
        directory = os.path.dirname(self.path_to_mp4)
        filename = os.path.basename(f"10 fun facts about {prompt_text}.mp4")
        
        pyautogui.hotkey('alt', 'd')
        pyautogui.write(directory)
        pyautogui.press('enter')
        self.random_time() 
        
        pyautogui.press('tab')
        self.random_time() 
        pyautogui.press('tab')
        self.random_time() 
        pyautogui.press('tab')
        self.random_time() 
        pyautogui.press('tab')
        self.random_time() 
        pyautogui.press('tab')
        self.random_time() 
        pyautogui.write(filename)
        self.random_time() 
        pyautogui.press('enter')

    def open_tiktok(self):
        """
        Open TikTok using Selenium with a specified user profile.
        """
        profile_path = r"path/to/your/chrome/user/profile" # Update this path accordingly
        options = Options()
        options.add_argument(f"--user-data-dir={profile_path}")
        options.add_argument("--profile-directory=Profile 1")

        self.driver = webdriver.Chrome(service=ChromeService(), options=options)
        
        stealth(self.driver,
                languages=["sv-SE", "sv"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True)

        time.sleep(2.2)
        self.driver.get("https://www.tiktok.com")
        self.random_time()

    def press_upload_video(self):
        """
        Press the upload video button on TikTok.
        """
        try:
            upload_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "div.css-1qup28j-DivUpload"))
            )
            upload_button.click()
            print("Upload button clicked")
        except Exception as e:
            print(f"An error occurred: {e}")

    def press_pick_video(self):
        """
        Press the button to pick the video file for upload.
        """
        try:
            iframe = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "iframe[data-tt='Upload_index_iframe']"))
            )
            self.driver.switch_to.frame(iframe)
            
            pick_video_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Välj video' and @data-hide='false']"))
            )
            pick_video_button.click()
            print("Pick video button pressed")
            self.driver.switch_to.default_content()
        except Exception as e:
            print(f"An error occurred: {e}")
            try:
                self.driver.switch_to.frame(iframe)
                element_html = self.driver.find_element(By.XPATH, "//button[@aria-label='Välj video' and @data-hide='false']").get_attribute('outerHTML')
                self.driver.switch_to.default_content()
                print(f"HTML of the element: {element_html}")
            except Exception as inner_exception:
                print(f"Could not retrieve element HTML: {inner_exception}")

    def write_video_info(self, post_text):
        """
        Write video information on TikTok.

        Args:
            post_text (str): Text to be added as video information.
        """
        post_text = f"10 interesting facts about {post_text}, from AI"
        try:
            iframe = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "iframe[src*='tiktok.com/creator']"))
            )
            self.driver.switch_to.frame(iframe)
            
            video_info_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#root > div > div > div > div.jsx-480949468.container-v2.form-panel.flow-opt-v1 > div > div.jsx-2745626608.form-v2.flow-opt-v1.reverse > div.jsx-2745626608.caption-wrap-v2 > div > div.jsx-4128635239.caption-markup > div.jsx-4128635239.caption-editor > div > div > div"))
            )
            video_info_field.click()
            
            self.driver.execute_script("""
                arguments[0].innerText = arguments[1];
                arguments[0].dispatchEvent(new Event('input', { bubbles: true }));
                arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
                arguments[0].dispatchEvent(new Event('blur'));
            """, video_info_field, post_text)
            
            video_info_field.click()
            self.driver.switch_to.default_content()
        except TimeoutException as e:
            print("Timeout while waiting for the video info input field to be clickable or present.")
            print(f"An error occurred: {e}")
        except Exception as e:
            print("An unexpected error occurred.")
            print(f"An error occurred: {e}")

    def press_extend_info_button(self):
        """
        Press the button to extend the information section on TikTok.
        """
        try:
            iframe = WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "iframe[src*='tiktok.com/creator']"))
            )
            self.driver.switch_to.frame(iframe)

            show_more_button = WebDriverWait(self.driver, 20).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "div.jsx-2745626608.more-btn"))
            )
            print("Show more button found")
            self.driver.execute_script("arguments[0].click();", show_more_button)
            print("Show more button clicked")
            self.driver.switch_to.default_content()
        except TimeoutException as e:
            print("Timeout while waiting for the show more button to be clickable or present.")
            print(f"An error occurred: {e}")
        except Exception as e:
            print("An unexpected error occurred.")
            print(f"An error occurred: {e}")

    def press_ai_content_button(self):
        """
        Press the button to mark the content as AI-generated on TikTok.
        """
        try:
            iframe = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "iframe[src*='tiktok.com/creator']"))
            )
            self.driver.switch_to.frame(iframe)
            
            switch_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/div/div/div/div[4]/div/div[2]/div[6]/div[2]/div[3]/div/div/div/div[1]/input"))
            )
            self.driver.execute_script("arguments[0].click();", switch_button)
            self.driver.switch_to.default_content()
        except TimeoutException as e:
            print("Timeout while waiting for the switch button to be clickable or present.")
            print(f"An error occurred: {e}")
        except Exception as e:
            print("An unexpected error occurred.")
            print(f"An error occurred: {e}")

    def press_publish(self):
        """
        Press the publish button to upload the video to TikTok.
        """
        try:
            iframe = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "iframe[src*='tiktok.com/creator']"))
            )
            self.driver.switch_to.frame(iframe)
            
            publish_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "/html/body/div[1]/div/div/div/div[4]/div/div[2]/div[8]/button[1]"))
            )
            self.driver.execute_script("arguments[0].click();", publish_button)
            self.driver.switch_to.default_content()
        except TimeoutException as e:
            print("Timeout while waiting for the publish button to be clickable or present.")
            print(f"An error occurred: {e}")
        except Exception as e:
            print("An unexpected error occurred.")
            print(f"An error occurred: {e}")

    def close_all(self):
        """
        Close the Selenium WebDriver.
        """
        if self.driver:
            self.driver.quit()

    def run(self, post_text):
        """
        Run the process of uploading a video to TikTok.

        Args:
            post_text (str): Text to use for video information.
        """
        self.open_tiktok()
        time.sleep(9)
        self.press_upload_video()
        self.random_time()
        self.press_pick_video()
        self.random_time()
        self.load_mp4(post_text)
        time.sleep(10)
        self.write_video_info(post_text)
        self.random_time()
        self.press_extend_info_button()
        self.random_time()
        self.press_ai_content_button()
        self.random_time()
        self.press_publish()
        time.sleep(5)

if __name__ == "__main__":
    mp4_to_tiktok = SendToTiktok()
    print("START")
    mp4_to_tiktok.run("China")
    print("END")
