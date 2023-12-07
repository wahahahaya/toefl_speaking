from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sounddevice as sd
import numpy as np


# 錄製參數
duration = 45  # 錄製10秒音頻
sampling_rate = 16000  # 採樣率

# 錄製音頻
print("開始錄音...")
audio = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1)
sd.wait()
print("錄音結束.")

# 將音頻轉換為numpy數組
audio_array = np.squeeze(audio)

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
model.config.forced_decoder_ids = None

input_features = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features 

# generate token ids
predicted_ids = model.generate(input_features)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
print(transcription)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)
