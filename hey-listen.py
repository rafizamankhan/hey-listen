import threading 
import subprocess 
import time 
import numpy as np 
import sounddevice as sd 
import whisper 
import rumps 
import pyperclip 
from pynput import keyboard

#configuration 
sample_rate = 16000
model_size = "base"
HOTKEY = {keyboard.Key.alt_l}

class AudioRecorder:

    def __init__(self, samplerate=sample_rate):
        self.samplerate = samplerate
        self.chunks = []
        self.recording = False
        self.stream = None

    def _callback(self, indata, frames, time_info, status):

        if self.recording:
            self.chunks.append(indata.copy())

    def start(self):
        self.chunks = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate = self.samplerate,
            channels=1,
            dtype='float32',
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

        if not self.chunks:
            return None
        
        audio = np.concatenate(self.chunks, axis=0).flatten()
        return audio



class WhisperTranscriber:

    def __init__(self, modelsize=model_size):
        print(f"Loading Whisper '{modelsize}' model...")
        self.model = whisper.load_model(modelsize)
        print("Model ready.")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) < sample_rate * 0.5:
            return "" 

        result = self.model.transcribe(
            audio,
            fp16=False,
            language='en',
            task="transcribe"
        )
        return result["text"].strip()

class HeyListen(rumps.App):

    def __init__(self):
        super().__init__("HeyListen", quit_button="Quit HeyListen")
        self.recorder = AudioRecorder()
        self.transcriber = WhisperTranscriber()
        self.pressed_keys = set()
        self.is_recording = False

        self.listener = keyboard.Listener(
            on_press = self._on_press,
            on_release = self._on_release
        )
        self.listener.start()
        self.title = "HeyListen"

    #Hotkey Logic
    def _normalize_key(self, key):
        if key == keyboard.Key.alt_r:
            return keyboard.Key.alt_l
        return key

    def _on_press(self, key):
        try:
            self.pressed_keys.add(self._normalize_key(key))
            if HOTKEY.issubset(self.pressed_keys) and not self.is_recording:
                self._start_recording()
        except Exception as e:
            print(f"Key press error: {e}")

    def _on_release(self, key):
        try:
            self.pressed_keys.discard(self._normalize_key(key))
            if self.is_recording and not HOTKEY.issubset(self.pressed_keys):
                self._stop_and_transcribe()
        except Exception as e:
            print(f"Key release error: {e}")

    def _start_recording(self):
        self.is_recording = True
        self.title = "Recording..."
        self.recorder.start()

    def _stop_and_transcribe(self):
        self.is_recording = False
        self.title = "Transcribing..."
        audio = self.recorder.stop()

        threading.Thread(target=self._transcribe_and_paste, args=(audio,), daemon=True).start()

    def _transcribe_and_paste(self, audio):
        text = self.transcriber.transcribe(audio)

        if text:

            text = " " + text
            pyperclip.copy(text)

            script = 'tell application "System Events" to keystroke "v" using command down'
            subprocess.run(["osascript", "-e", script])

            print(f"Transcribed: {text}")
            self.title = "Success"

        else:
            print("No speech detected")
            self.title = "Error 404"

        time.sleep(2)
        self.title = "HeyListen"
        





if __name__ == "__main__":
    hl_application = HeyListen()
    hl_application.run()