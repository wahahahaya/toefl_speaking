import sounddevice as sd
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf


def record_audio(duration=10, sample_rate=16000):
    """Record audio from the microphone for a given duration in mono."""
    print("Recording...")
    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
        sd.wait()
        sf.write("recording.wav", recording, sample_rate)
    except Exception as e:
        print("An error occurred while recording:", e)
        return None
    print("Recording stopped.")
    return recording


def transcribe_audio(audio_data):
    if audio_data is None:
        return "No audio data to transcribe."

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

    inputs = processor(audio_data, return_tensors="pt", sampling_rate=16000)
    predicted_ids = model.generate(inputs.input_values)
    result = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return result


def main():
    # Record audio for 45 seconds
    audio_data = record_audio()

    # Transcribe the recorded audio
    if audio_data is not None:
        transcription = transcribe_audio(audio_data)
        print("Transcription:\n", transcription)
    else:
        print("Failed to record audio.")

if __name__ == "__main__":
    main()
