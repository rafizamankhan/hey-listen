import threading 
import subprocess 
import time 
import queue
import objc
import numpy as np 
import sounddevice as sd 
import whisper 
import rumps 
import pyperclip 
from pynput import keyboard
from AppKit import (NSWindow, NSView, NSColor, NSBezierPath, NSScreen,
                    NSFloatingWindowLevel, NSBackingStoreBuffered)
from Foundation import NSObject, NSTimer

#configuration 
sample_rate = 16000
model_size = "base"
HOTKEY = {keyboard.Key.alt_l}

class _OverlayWindow(NSWindow):
    def canBecomeKeyWindow(self):
        return False
    def canBecomeMainWindow(self):
        return False


class _PillView(NSView):

    def initWithFrame_(self, frame):
        self = objc.super(_PillView, self).initWithFrame_(frame)
        if self is not None:
            self.amps = [0.0] * 10
            self.active = False
        return self

    def isOpaque(self):
        return False

    def drawRect_(self, rect):
        bounds = self.bounds()
        w, h = bounds.size.width, bounds.size.height
        pill = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(bounds, h / 2, h / 2)

        if self.active:
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0, 0, 0, 0.8).setFill()
            pill.fill()
            margin = h * 0.5
            usable_w = w - 2 * margin
            bar_w = usable_w / len(self.amps)
            gap = bar_w * 0.3
            cap_w = bar_w - gap
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.9).setFill()
            for i, amp in enumerate(self.amps):
                bar_h = max(amp * (h - 10), 3)
                x = margin + i * bar_w + gap / 2
                y = (h - bar_h) / 2
                r = min(cap_w, bar_h) / 2
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    ((x, y), (cap_w, bar_h)), r, r
                ).fill()
        else:
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0, 0, 0, 0.35).setFill()
            pill.fill()
            dot = 4
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.5).setFill()
            NSBezierPath.bezierPathWithOvalInRect_(
                (((w - dot) / 2, (h - dot) / 2), (dot, dot))
            ).fill()


IDLE_W, IDLE_H = 20, 12
ACTIVE_W, ACTIVE_H = 60, 30
HUD_Y = 90


class WaveformHUD(NSObject):

    @objc.python_method
    def setup(self):
        self._queue = queue.Queue()
        self._timer = None
        self._win = None
        self._view = None
        self.performSelector_withObject_afterDelay_('initWindow:', None, 0)
        return self

    def initWindow_(self, sender):
        self._create_window()
        self._win.orderFront_(None)

    @objc.python_method
    def _create_window(self):
        if self._win is not None:
            return
        scr = NSScreen.mainScreen().frame()
        w, h = IDLE_W, IDLE_H
        self._win = _OverlayWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            (((scr.size.width - w) / 2, HUD_Y), (w, h)),
            0,
            NSBackingStoreBuffered,
            False
        )
        self._win.setLevel_(NSFloatingWindowLevel)
        self._win.setOpaque_(False)
        self._win.setBackgroundColor_(NSColor.clearColor())
        self._win.setHasShadow_(False)
        self._win.setIgnoresMouseEvents_(True)
        self._win.setCollectionBehavior_(1 | 16)
        self._view = _PillView.alloc().initWithFrame_(((0, 0), (w, h)))
        self._view.setAutoresizingMask_(2 | 16)
        self._win.setContentView_(self._view)

    @objc.python_method
    def show(self):
        self._queue = queue.Queue()
        self.performSelectorOnMainThread_withObject_waitUntilDone_('doShow:', None, False)

    def doShow_(self, sender):
        if self._win is None:
            self._create_window()
        if self._timer:
            self._timer.invalidate()
            self._timer = None
        self._view.active = True
        self._view.amps = [0.0] * 10
        scr = NSScreen.mainScreen().frame()
        w, h = ACTIVE_W, ACTIVE_H
        self._win.setFrame_display_animate_(
            (((scr.size.width - w) / 2, HUD_Y), (w, h)), True, True
        )
        self._win.orderFront_(None)
        self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.05, self, 'tick:', None, True
        )

    @objc.python_method
    def hide(self):
        self.performSelectorOnMainThread_withObject_waitUntilDone_('doHide:', None, False)

    def doHide_(self, sender):
        if self._timer:
            self._timer.invalidate()
            self._timer = None
        if self._win is None:
            return
        self._view.active = False
        scr = NSScreen.mainScreen().frame()
        w, h = IDLE_W, IDLE_H
        self._win.setFrame_display_animate_(
            (((scr.size.width - w) / 2, HUD_Y), (w, h)), True, True
        )
        self._view.setNeedsDisplay_(True)

    @objc.python_method
    def feed(self, chunk):
        self._queue.put(chunk)

    def tick_(self, timer):
        while not self._queue.empty():
            chunk = self._queue.get_nowait()
            rms = np.sqrt(np.mean(chunk ** 2))
            self._view.amps.pop(0)
            self._view.amps.append(min(rms * 10, 1.0))
        self._view.setNeedsDisplay_(True)


class AudioRecorder:

    def __init__(self, samplerate=sample_rate):
        self.samplerate = samplerate
        self.chunks = []
        self.recording = False
        self.stream = None
        self.on_audio = None

    def _callback(self, indata, frames, time_info, status):

        if self.recording:
            self.chunks.append(indata.copy())
            if self.on_audio:
                self.on_audio(indata.copy())

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
        self.hud = WaveformHUD.alloc().init()
        self.hud.setup()
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
        self.hud.show()
        self.recorder.on_audio = self.hud.feed
        self.recorder.start()

    def _stop_and_transcribe(self):
        self.is_recording = False
        self.pressed_keys.clear()
        self.title = "Transcribing..."
        audio = self.recorder.stop()
        self.recorder.on_audio = None
        self.hud.hide()

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