import sounddevice as sd
import numpy as np 
import whisper 

model = whisper.load_model("base") 

sample_rate = 16000
duration = 10

print("hey listen")
audio = sd.rec(frames=int(duration*sample_rate), samplerate=sample_rate, channels=1, dtype='float32', )
sd.wait()
audio = audio.flatten()

result = model.transcribe(audio, fp16=False)

print(f"\nTranscribed text: {result['text']}")
print(f"\nLanguage detected: {result['language']}")

for seg in result['segments']:
    print(f" [{seg['start']:.1f}s -> {seg['end']:.1f}s] {seg['text']}")

