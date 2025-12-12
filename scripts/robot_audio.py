import sounddevice as sd
import numpy as np
import webrtcvad
import time
import base64
import requests
import os
import resampy
import queue
import threading

AZURE_WHISPER_ENDPOINT = os.getenv("AZURE_WHISPER_ENDPOINT")
AZURE_WHISPER_API_KEY = os.getenv("AZURE_WHISPER_API_KEY")

ORIGINAL_SR = 48000
TARGET_SR = 16000
FRAME_DURATION = 30  # ms
ORIGINAL_FRAME_SAMPLES = int(ORIGINAL_SR * FRAME_DURATION / 1000)  # 1440
TARGET_FRAME_SAMPLES = int(TARGET_SR * FRAME_DURATION / 1000)      # 480
VAD_LEVEL = 2
SILENCE_TIME = 0.7

vad = webrtcvad.Vad(VAD_LEVEL)
audio_queue = queue.Queue()
buffer = bytearray()
last_voice = time.time()

# List available input devices
print("Available input devices:\n")
for i, dev in enumerate(sd.query_devices()):
    if dev['max_input_channels'] > 0:
        print(i, dev['name'], dev['max_input_channels'], dev['hostapi'])

MIC_DEVICE_INDEX = 5  # change as needed

# -----------------------------
# Whisper transcription
# -----------------------------
def transcribe_with_whisper(audio_bytes):
    print("DEBUG: Sending segment to Whisper...")
    encoded = base64.b64encode(audio_bytes).decode("utf-8")
    payload = {
        "model": "whisper",
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded,
                            "format": "wav"
                        }
                    }
                ]
            }
        ]
    }
    headers = {"Content-Type": "application/json",
               "api-key": AZURE_WHISPER_API_KEY}
    resp = requests.post(AZURE_WHISPER_ENDPOINT, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"][0]["text"]
    print("DEBUG: Whisper returned text:", text)
    return text

# -----------------------------
# Callback: just queue audio
# -----------------------------
def callback(indata, frames, time_info, status):
    if status:
        print("DEBUG: InputStream status:", status)
    # downmix to mono
    mono = indata.mean(axis=1)
    # copy to queue
    audio_queue.put(mono.copy())

# -----------------------------
# Worker thread: process audio
# -----------------------------
def process_audio():
    global buffer, last_voice
    resample_buffer = np.array([], dtype=np.float32)

    while True:
        chunk = audio_queue.get()  # blocking
        resample_buffer = np.concatenate((resample_buffer, chunk))
        print(f"DEBUG: Resample buffer length: {len(resample_buffer)}")

        # process in ORIGINAL_FRAME_SAMPLES chunks
        while len(resample_buffer) >= ORIGINAL_FRAME_SAMPLES:
            frame = resample_buffer[:ORIGINAL_FRAME_SAMPLES]
            resample_buffer = resample_buffer[ORIGINAL_FRAME_SAMPLES:]

            # resample to TARGET_SR
            mono_16k = resampy.resample(frame, ORIGINAL_SR, TARGET_SR)
            if len(mono_16k) != TARGET_FRAME_SAMPLES:
                print(f"WARNING: Resampled frame has {len(mono_16k)} samples instead of {TARGET_FRAME_SAMPLES}")
            pcm = (mono_16k * 32768).astype(np.int16).tobytes()

            try:
                is_speech = vad.is_speech(pcm, TARGET_SR)
            except Exception as e:
                print("ERROR: VAD processing failed:", e)
                continue

            print(f"DEBUG: VAD result: {'speech' if is_speech else 'silence'}")
            if is_speech:
                buffer.extend(pcm)
                last_voice = time.time()
            else:
                if len(buffer) > 0 and (time.time() - last_voice) > SILENCE_TIME:
                    segment = bytes(buffer)
                    buffer.clear()
                    on_utterance_complete(segment)

# -----------------------------
# Handle completed utterance
# -----------------------------
def on_utterance_complete(segment):
    print("\n⏳ Transcribing...")
    try:
        text = transcribe_with_whisper(segment)
        print("🗣  Sentence:", text)
    except Exception as e:
        print("❌ Error during transcription:", e)

# -----------------------------
# Main loop
# -----------------------------
def main():
    print("🎤 Listening from mic... Ctrl+C to stop.")
    # start worker thread
    threading.Thread(target=process_audio, daemon=True).start()

    with sd.InputStream(
        device=MIC_DEVICE_INDEX,
        channels=2,
        samplerate=ORIGINAL_SR,
        dtype='float32',
        callback=callback
    ):
        while True:
            time.sleep(0.1)

if __name__ == "__main__":
    main()
