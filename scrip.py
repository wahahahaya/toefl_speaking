from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sounddevice as sd
import soundfile as sf
import numpy as np


# parameters
duration = 45  # total recording duration
sampling_rate = 16000  # sampling rate of audio
segment_length = 10  # length of each segment in seconds

def record(duration, sampling_rate):
    print("開始錄音...")
    audio = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1)
    sd.wait()
    sf.write("recording.wav", audio, sampling_rate)
    print("錄音結束.")
    return audio

def transcribe(audio):
    # split audio into segments
    audio_array = np.squeeze(audio)
    segments = np.array_split(audio_array, np.ceil(duration / segment_length))

    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
    model.config.forced_decoder_ids = None

    transcriptions = []
    for segment in segments:
        input_features = processor(segment, sampling_rate=sampling_rate, return_tensors="pt").input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        transcriptions.append(transcription[0])
    return transcriptions

def clean_transcription(transcriptions):
    cleaned_transcriptions = [sentence.strip() for sentence in transcriptions]
    full_text = " ".join(cleaned_transcriptions)
    return full_text


def main():
    audio = record(duration, sampling_rate)

    # transcribe audio
    if audio is not None:
        transcriptions = transcribe(audio)
        full_text = clean_transcription(transcriptions)
        print(full_text)
    else:
        print("No audio was recorded.")

if __name__ == "__main__":
    main()
