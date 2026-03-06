import threading 
import subprocess 
import time 
import queue
import math
import random
import objc
import numpy as np 
import sounddevice as sd 
import whisper 
import rumps 
import pyperclip 
import httpx
from pynput import keyboard
from AppKit import (NSWindow, NSView, NSColor, NSBezierPath, NSScreen,
                    NSFloatingWindowLevel, NSBackingStoreBuffered)
from Foundation import NSObject, NSTimer

#configuration 
sample_rate = 16000
model_size = "base"
HOTKEY = {keyboard.Key.alt_l}

APP_MODE_MAP = {
    "Google Docs":          "essay",
    "Microsoft Word":       "essay",
    "Pages":                "essay",
    "TextEdit":             "essay",
    "Notion":               "bullets",
    "Obsidian":             "bullets",
    "Bear":                 "bullets",
    "Notes":                "bullets",
    "Mail":                 "email",            
    "Microsoft Outlook":    "email",
    "Gmail":                "email",
    "Slack":                "message",
    "WhatsApp":              "message",
    "Messages":              "message"
}

#Signal
essay_signals = ["essay", "thesis", "argument", "paragraph", "introduction", "conclusion", "report", "paper", "write about", "main point", "claim"]
bullet_signals = ["bullet", "bullets", "list", "notes", "key points", "summarize", "action items", "to do", "todo", "tasks", "points", "breakdown"]
email_signals = ["email", "write to", "send to", "reply", "dear", "subject line", "regards"]
message_signals = ["text", "message", "slack", "dm", "respond", "comments"]

TASK_SIGNALS = [
    "write", "draft", "create", "make", "compose",
    "turn this into", "structure", "format",
    "email to", "message to", "text to", "send to",
    "reply to", "respond to",
    "essay about", "outline for", "notes on",
    "bullet points", "summarize",
]

def is_task(transcript: str) -> bool:
    t = transcript.lower().strip()
    return any(signal in t for signal in TASK_SIGNALS)

#Find Active Application
def get_active_app() -> str:
    try:
        script = 'tell application "System Events" to get name of first process whose frontmost is true'
        r = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
        return r.stdout.strip()
    except Exception:
        return "Unknown"

#Detect Application Mode
def detect_mode(transcript: str) -> str:
    t = transcript.lower()

    if any(w in t for w in bullet_signals): return "bullets"
    if any(w in t for w in essay_signals): return "essay"
    if any(w in t for w in email_signals): return "email"
    if any(w in t for w in message_signals): return "message"

    if len(t.split()) > 15:
        app = get_active_app()
        return APP_MODE_MAP.get(app, "clean")

    return "clean"

#Prompts for LLM (Phi3.5)
PROMPTS = {
        "essay": """You are an academic writing assistant. The user spoke their thoughts out loud.
    Transform the transcript into a structured essay outline with:
    - A clear thesis statement
    - 2-3 numbered main points with brief supporting points
    - One sentence closing thought

    Be concise. Use the user's own words and ideas. Do not add information they did not mention.

    Transcript: {transcript}

    Output only the structured outline. No preamble.""",

        "bullets": """You are a note taking assistant. The user spoke their thoughts out loud.
    Transform the transcript into clean bullet points:
    - Group related ideas together
    - Use short, clear phrases
    - Only include bullet points that are related to the transcript 
    - Do not repeat ideas and keep bullet points to minimum 
    - Maximum 8 bullets per group (Does not mean you need to keep reiterating current points to fill it up to 8 points)

    Transcript: {transcript}

    Output only the structured outline. No preamble.""",

        "email": """You are an email writing assistant. The user spoke their thoughts out loud. 
    Transform the transcript into a professional email with:
    - Subject: [subject line]
    - A clear opening line
    - 1-2 concise body paragraphs
    - A professional sign-off

    Transcript: {transcript}

    Output only the structured outline. No preamble.""",

        "message": """You are a messaging assistant. The user spoke their thoughts out loud. 
    Transform the transcript into a natural, conversational message.
    Remove filler words. Keep the tone friendly and direct.
    Keep it to 2-3 sentences maximum. 

    Transcript: {transcript}

    Output only the structured outline. No preamble.""",

        "clean": """You are transcription cleaner. The user spoke out their thoughts our loud. 
    Clean up the transcript by:
    - Removing filler words like um, uh, like, you know 
    - Fixing run-on sentences
    - Keeping all original ideas and wording

    Transcript: {transcript}

    Output only the cleaned text. No preamble.""",
}

# phi3.5 intelligence layer
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3.5:latest"

_session_context: list[str] = []

def pre_warm_ollama():
    try: 
        httpx.post (
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": "Hey Listen",
                "stream": True,
                "options": {"num_predict": 1}
            },
            timeout = 60.0
            )
        print(f"Phi3.5 warmed and ready")
    except httpx.ConnectError:
        print("Ollama not running - falling back to raw transcript")
    except Exception  as e:
        print(f"Phi3.5 pre-warm failed: {e}")
 
def listen_with_phi(transcript: str, mode: str) -> str:
    """Send transcript to local Phi3.5 via Ollama. Falls back to raw transcript. """
    global _session_context 

    #Context prefix for essay mode
    context_prefix = ""
    if mode == "essay" and _session_context:
        context_prefix = "Previous notes in this session:\n"
        context_prefix += "\n---\n".join(_session_context[-3:])
        context_prefix += "\n\nNew transcript to structure:\n"

    prompt = PROMPTS[mode].format(
        transcript = context_prefix + transcript
    )

    try:
        response = httpx.post (
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 250,
                    "stop": ["\n\n\n"],
                    "repeat_penalty": 1.15
                }
            },
            timeout=30.0
        )
        response.raise_for_status()
        result = response.json()["response"].strip()

        #context memory
        _session_context.append(transcript)
        if len(_session_context) > 10: 
            _session_context.pop(0)
        return result

    except httpx.ConnectError:
        print("Ollama not running - falling back to raw transcript")
        return transcript
    except Exception  as e:
        print(f"Phi3.5 error: {e} - falling back to raw transcript")
        return transcript

def clear_session():
    global _session_context
    _session_context = []
    print("Session cleared.")

#overlay 
class _OverlayWindow(NSWindow):
    def canBecomeKeyWindow(self):
        return False
    def canBecomeMainWindow(self):
        return False

#pillview(UI) 
class _PillView(NSView):

    def initWithFrame_(self, frame):
        self = objc.super(_PillView, self).initWithFrame_(frame)
        if self is not None:
            self.amps = [0.0] * NUM_BARS
            self.active = False
            self._transition_t = 0.0
        return self

    def isOpaque(self):
        return False

    def drawRect_(self, rect):
        bounds = self.bounds()
        w, h = bounds.size.width, bounds.size.height
        pill = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(bounds, h / 2, h / 2)

        if self.active:
            p = RING_PAD
            NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, 0.12).setFill()
            pill.fill()

            inner_bounds = ((p, p), (w - 2 * p, h - 2 * p))
            inner_h = h - 2 * p
            inner_w = w - 2 * p
            inner_pill = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                inner_bounds, inner_h / 2, inner_h / 2
            )
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0, 0, 0, 0.99).setFill()
            inner_pill.fill()

            margin = inner_w * 0.15
            usable_w = inner_w - 2 * margin
            bar_w = usable_w / len(self.amps)
            cap_w = max(bar_w * 0.40, 2)
            bloom_pad = 1.0
            center_idx = (len(self.amps) - 1) / 2
            for i, amp in enumerate(self.amps):
                delay = abs(i - center_idx) / center_idx * 0.3
                t = max(min((self._transition_t - delay) / (1.0 - delay), 1.0), 0.0)
                bar_h = max(amp * inner_h * 0.58 * t, 3.0 * t)
                x = p + margin + i * bar_w + (bar_w - cap_w) / 2
                y = p + (inner_h - bar_h) / 2
                r = cap_w / 2
                a = EDGE_ALPHA[i]
                NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, a * 0.25).setFill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    ((x - bloom_pad, y - bloom_pad),
                     (cap_w + bloom_pad * 2, bar_h + bloom_pad * 2)),
                    r + bloom_pad, r + bloom_pad
                ).fill()
                NSColor.colorWithCalibratedRed_green_blue_alpha_(1, 1, 1, a * 0.85).setFill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    ((x, y), (cap_w, bar_h)), r, r
                ).fill()
        else:
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0, 0, 0, 0.65).setFill()
            pill.fill()

#pill-specs
IDLE_W, IDLE_H = 32, 12
ACTIVE_W, ACTIVE_H = 65, 25
HUD_Y = 85
NUM_BARS = 9
RING_PAD = 1.5

def _make_bell(n):
    center = (n - 1) / 2
    return [math.exp(-0.5 * ((i - center) / (n / 3.5)) ** 2) for i in range(n)]

def _make_edge_alpha(n):
    center = (n - 1) / 2
    return [0.25 + 0.75 * math.exp(-0.5 * ((i - center) / (n / 2.8)) ** 2) for i in range(n)]

def _make_smoothing(n):
    center = (n - 1) / 2
    return [0.25 + 0.2 * abs(i - center) / center for i in range(n)]

BELL = _make_bell(NUM_BARS)
EDGE_ALPHA = _make_edge_alpha(NUM_BARS)
SMOOTHING = _make_smoothing(NUM_BARS)

#HUD(UI)
class WaveformHUD(NSObject):

    @objc.python_method
    def setup(self):
        self._queue = queue.Queue()
        self._timer = None
        self._win = None
        self._view = None
        self._phase = 0.0
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
        self._view.amps = [0.0] * NUM_BARS
        self._view._transition_t = 0.0
        scr = NSScreen.mainScreen().frame()
        win_w = ACTIVE_W + RING_PAD * 2
        win_h = ACTIVE_H + RING_PAD * 2

        self._win.setAlphaValue_(0.4)
        self._win.setFrame_display_animate_(
            (((scr.size.width - win_w) / 2, HUD_Y - RING_PAD), (win_w, win_h)),
            True, True
        )
        self._win.setAlphaValue_(1.0)
        self._win.orderFront_(None)

        self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.033, self, 'tick:', None, True
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

        self._win.setAlphaValue_(0.85)
        self._win.setFrame_display_animate_(
            (((scr.size.width - w) / 2, HUD_Y), (w, h)), True, True
        )
        self._win.setAlphaValue_(1.0)
        self._view.setNeedsDisplay_(True)

    @objc.python_method
    def feed(self, chunk):
        self._queue.put(chunk)

    def tick_(self, timer):
        if self._view._transition_t < 1.0:
            self._view._transition_t = min(self._view._transition_t + 0.08, 1.0)
        self._phase += 0.32

        rms = 0.0
        while not self._queue.empty():
            chunk = self._queue.get_nowait()
            rms = max(rms, np.sqrt(np.mean(chunk ** 2)))
        rms = min(rms * 10, 1.0)

        for i in range(NUM_BARS):
            wave = 0.5 + 0.5 * math.sin(self._phase - i * 0.7)
            idle_pulse = wave * 0.31

            jitter = 0.7 + random.random() * 0.6
            voice = rms * BELL[i] * jitter
            target = max(voice, idle_pulse)

            s = SMOOTHING[i]
            self._view.amps[i] = self._view.amps[i] * (1 - s) + target * s
        self._view.setNeedsDisplay_(True)

#record logic
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

def pre_warm_whisper(transcriber):
    transcriber.transcribe(np.zeros(sample_rate, dtype=np.float32))
    print("Whisper warmed and ready")

#transcribe logic
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
            temperature = 0,
            condition_on_previous_text = False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            task='transcribe'
        )

        '''
            verbose = False,
            hallucination_silence_threshold=True,
            
        '''
        return result["text"].strip()

#voice activity detection


#application
class HeyListen(rumps.App):

    def __init__(self):
        super().__init__("HeyListen", quit_button="Quit HeyListen")
        self.recorder = AudioRecorder()
        self.transcriber = WhisperTranscriber()
        self.hud = WaveformHUD.alloc().init()
        self.hud.setup()
        self.pressed_keys = set()
        self.is_recording = False
        self._press_time = 0
        self._last_tap_time = 0
        self._locked = False

        self.listener = keyboard.Listener(
            on_press = self._on_press,
            on_release = self._on_release
        )
        self.listener.start()
        
        self.title = "HeyListen"
        threading.Thread(target=pre_warm_whisper, args=(self.transcriber,), daemon=True).start()
        threading.Thread(target=pre_warm_ollama, daemon=True).start()

        self.menu = [
            rumps.MenuItem("Clear Session", callback = self._clear_session),
            None,
        ]

    #Hotkey Logic
    def _normalize_key(self, key):
        if key == keyboard.Key.alt_r:
            return keyboard.Key.alt_l
        return key

    def _on_press(self, key):
        try:
            self.pressed_keys.add(self._normalize_key(key))
            
            if not HOTKEY.issubset(self.pressed_keys):
                return

            self._press_time = time.time()

            if self._locked and self.is_recording:
                self._stop_and_transcribe()
                self._locked = False
                return
            
            if not self.is_recording:
                self._start_recording()

        except Exception as e:
            print(f"Key press error: {e}")

    def _on_release(self, key):
        try:
            self.pressed_keys.discard(self._normalize_key(key))

            if self._locked: 
                return

            if not self.is_recording:
                return
            
            held = time.time() - self._press_time

            if held < 0.25:
                now = time.time()
                if now - self._last_tap_time < 0.4:
                    self._locked = True
                    self._last_tap_time = 0
                    return
                else:
                    self._last_tap_time = now
                    self._stop_and_transcribe()
            else:
                if not HOTKEY.issubset(self.pressed_keys):
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
        self.title = "Transcribing..."
        audio = self.recorder.stop()
        self.recorder.on_audio = None
        self.hud.hide()

        threading.Thread(target=self._transcribe_and_paste, args=(audio,), daemon=True).start()

    def _transcribe_and_paste(self, audio):
        text = self.transcriber.transcribe(audio)

        if text:
            if is_task(text):
                mode = detect_mode(text)
                print(f"Task detected → mode: {mode}")
                self.title = "Thinking..."
                output = " " + listen_with_phi(text, mode)
                print(f"Transcribed: {text}")
                print(f"Structured ({mode}): {output}")
            else:
                output = " " + text
                print(f"Transcribed: {text}")

            pyperclip.copy(output)
            script = 'tell application "System Events" to keystroke "v" using command down'
            subprocess.run(["osascript", "-e", script])
            self.title = "Success"

        else:
            print("No speech detected")
            self.title = "Error 404"

        time.sleep(2)
        self.title = "HeyListen"

    def _clear_session(self, _):
        clear_session()
        print(f"Listen", "Session cleared", "Starting fresh context.")

#main
if __name__ == "__main__":
    hl_application = HeyListen()
    hl_application.run()