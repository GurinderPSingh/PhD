import os
import subprocess
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

# Use a raw string to handle backslashes
video_path = r"E:\carleton\ELEC-5401-F\Lectures\Lecture_21.mp4"
audio_output_dir = r"C:\Temp"
audio_path = os.path.join(audio_output_dir, "output_audio.wav")

if not os.path.exists(video_path):
    print(f"File not found: {video_path}")
    exit(1)

if not os.path.exists(audio_output_dir):
    os.makedirs(audio_output_dir)

print("Extracting audio using FFmpeg...")
try:
    subprocess.run(
        [
            "ffmpeg",
            "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", audio_path
        ],
        check=True
    )
    print("Audio extracted successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error during audio extraction: {e}")
    exit(1)

# Split audio into smaller chunks and check for blank audio
print("Splitting audio into smaller chunks...")
audio = AudioSegment.from_file(audio_path)
chunk_length_ms = 60000  # 1 minute
chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

for idx, chunk in enumerate(chunks):
    # Detect non-silent parts to skip blank audio
    nonsilent_ranges = detect_nonsilent(chunk, min_silence_len=1000, silence_thresh=-40)
    if nonsilent_ranges:
        chunk.export(os.path.join(audio_output_dir, f"chunk_{idx}.wav"), format="wav")
    else:
        print(f"Skipping blank chunk_{idx}.wav")

# Transcribe each chunk
print("Transcribing audio...")
recognizer = sr.Recognizer()
for file in os.listdir(audio_output_dir):
    if file.startswith("chunk_") and file.endswith(".wav"):
        print(f"Transcribing {file}...")
        try:
            with sr.AudioFile(os.path.join(audio_output_dir, file)) as source:
                recognizer.adjust_for_ambient_noise(source)
                audio_content = recognizer.record(source)
                transcription = recognizer.recognize_google(audio_content)
                with open(os.path.join(audio_output_dir, "lecture_transcription.txt"), "a") as f:
                    f.write(transcription + "\n")
        except Exception as e:
            print(f"Error during transcription of {file}: {e}")
