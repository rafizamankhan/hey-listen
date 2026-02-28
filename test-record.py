import sounddevice as sd
import numpy as np 

sample_rate = 16000
duration = 10

print("Recording for 10 seconds...")

audio = sd.rec(

    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype='float32'

)
sd.wait()

audio = audio.flatten()
print(f"Captured {len(audio)} samples")
print(f"Max amplitude: {np.max(np.abs(audio)):.4f}")
print(f"Audio array (first 10): {audio[:10]}")
