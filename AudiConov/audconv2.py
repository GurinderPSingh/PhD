import os
import subprocess
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import whisper
import tkinter as tk
from tkinter import filedialog

# Function to process each lecture
def process_lecture(video_path, output_dir):
    lecture_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{lecture_name}.wav")
    transcription_path = os.path.join(output_dir, f"{lecture_name}.txt")

    print(f"Processing lecture: {lecture_name}")

    # Step 1: Extract audio
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
        print(f"Audio extracted successfully for {lecture_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error during audio extraction for {lecture_name}: {e}")
        return

    # Step 2: Split audio into smaller chunks
    print("Splitting audio into smaller chunks...")
    audio = AudioSegment.from_file(audio_path)
    chunk_length_ms = 60000  # 1 minute
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    chunk_files = []
    for idx, chunk in enumerate(chunks):
        nonsilent_ranges = detect_nonsilent(chunk, min_silence_len=1000, silence_thresh=-40)
        if nonsilent_ranges:
            chunk_file = os.path.join(output_dir, f"chunk_{idx}.wav")
            chunk.export(chunk_file, format="wav")
            chunk_files.append(chunk_file)
        else:
            print(f"Skipping blank chunk_{idx}.wav")

    # Step 3: Transcribe each chunk using Whisper
    print("Transcribing audio with Whisper...")
    model = whisper.load_model("base")  # Use "small", "medium", or "large" as needed.

    with open(transcription_path, "w") as f:
        for chunk_file in chunk_files:
            print(f"Transcribing {os.path.basename(chunk_file)} with Whisper...")
            try:
                result = model.transcribe(chunk_file)
                f.write(result["text"] + "\n")
            except Exception as e:
                print(f"Error during Whisper transcription of {os.path.basename(chunk_file)}: {e}")

    print(f"Transcription saved as: {transcription_path}")

    # Step 4: Clean up chunks
    print("Deleting audio chunks...")
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    print(f"Chunks deleted for {lecture_name}")

# Function to browse and queue lectures
def browse_lectures():
    files = filedialog.askopenfilenames(
        title="Select Lecture Files",
        filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")]
    )
    for file in files:
        lecture_listbox.insert(tk.END, file)

# Function to start processing the queue
def start_processing():
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    if not output_dir:
        print("No output directory selected.")
        return

    for idx in range(lecture_listbox.size()):
        video_path = lecture_listbox.get(idx)
        process_lecture(video_path, output_dir)

    print("All lectures processed successfully!")

# Create the GUI
root = tk.Tk()
root.title("Lecture Processor")

frame = tk.Frame(root)
frame.pack(pady=10)

lecture_listbox = tk.Listbox(frame, width=80, height=20)
lecture_listbox.pack(side=tk.LEFT, padx=10)

scrollbar = tk.Scrollbar(frame, orient="vertical", command=lecture_listbox.yview)
scrollbar.pack(side=tk.RIGHT, fill="y")

lecture_listbox.config(yscrollcommand=scrollbar.set)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

browse_button = tk.Button(button_frame, text="Browse Lectures", command=browse_lectures)
browse_button.pack(side=tk.LEFT, padx=10)

process_button = tk.Button(button_frame, text="Start Processing", command=start_processing)
process_button.pack(side=tk.LEFT, padx=10)

quit_button = tk.Button(button_frame, text="Quit", command=root.quit)
quit_button.pack(side=tk.LEFT, padx=10)

root.mainloop()
