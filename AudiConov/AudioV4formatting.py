import os
import subprocess
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import whisper
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk  # For progress bars
import threading
import openai  # For transcript formatting with ChatGPT

# Set your OpenAI API key (or ensure it is set in your environment)
openai.api_key = os.getenv("OPENAI_API_KEY")


def reformat_transcript(raw_transcript, progress_callback):
    """
    Uses the ChatGPT API to reformat a raw transcript into a more organized and formatted version.
    """
    prompt = (
        "Please reformat the following transcript into a clean, organized, and well-structured format. "
        "Include punctuation, proper paragraph breaks, and any necessary headings if applicable:\n\n"
        f"{raw_transcript}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert transcription editor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        formatted_text = response.choices[0].message.content.strip()
        progress_callback(95)
        return formatted_text
    except Exception as e:
        print(f"Error during transcript formatting: {e}")
        return raw_transcript


def process_lecture(video_path, output_dir, model_choice, organize_transcript, google_language, progress_callback):
    lecture_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{lecture_name}.wav")
    transcription_path = os.path.join(output_dir, f"{lecture_name}.txt")
    formatted_transcription_path = os.path.join(output_dir, f"{lecture_name}_formatted.txt")

    print(f"Processing lecture: {lecture_name}")

    # Step 1: Extract audio using FFmpeg
    progress_callback(0)
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
        progress_callback(20)
    except subprocess.CalledProcessError as e:
        print(f"Error during audio extraction for {lecture_name}: {e}")
        return

    # Step 2: Split audio into smaller chunks
    print("Splitting audio into smaller chunks...")
    audio = AudioSegment.from_file(audio_path)
    chunk_length_ms = 60000  # 1 minute chunks
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    chunk_files = []
    for idx, chunk in enumerate(chunks):
        nonsilent_ranges = detect_nonsilent(chunk, min_silence_len=1000, silence_thresh=-40)
        if nonsilent_ranges:
            chunk_file = os.path.join(output_dir, f"{lecture_name}_chunk_{idx}.wav")
            chunk.export(chunk_file, format="wav")
            chunk_files.append(chunk_file)
        else:
            print(f"Skipping blank chunk_{idx}.wav")
    progress_callback(40)

    # Step 3: Transcribe each chunk
    print("Transcribing audio...")
    # Open file with UTF-8 encoding to handle non-ASCII characters.
    with open(transcription_path, "w", encoding="utf-8") as f:
        total_chunks = len(chunk_files)
        if model_choice == "OpenAI Whisper":
            model = whisper.load_model("base")  # Choose model size as needed.
            for i, chunk_file in enumerate(chunk_files):
                print(f"Transcribing {os.path.basename(chunk_file)} with Whisper...")
                try:
                    result = model.transcribe(chunk_file)
                    f.write(result["text"] + "\n")
                except Exception as e:
                    print(f"Error during Whisper transcription of {os.path.basename(chunk_file)}: {e}")
                progress = 40 + int((i + 1) / total_chunks * 40)
                progress_callback(progress)
        elif model_choice == "Google Web Speech API":
            recognizer = sr.Recognizer()
            for i, chunk_file in enumerate(chunk_files):
                print(f"Transcribing {os.path.basename(chunk_file)} with Google API...")
                try:
                    with sr.AudioFile(chunk_file) as source:
                        audio_data = recognizer.record(source)
                    # Use the provided language code: "pa-IN" for Punjabi, "en-US" for English.
                    text = recognizer.recognize_google(audio_data, language=google_language)
                    f.write(text + "\n")
                except sr.UnknownValueError:
                    print(f"Google Speech API could not understand {os.path.basename(chunk_file)}")
                    f.write("[Unrecognized]\n")
                except sr.RequestError as e:
                    print(f"Google Speech API error for {os.path.basename(chunk_file)}: {e}")
                    f.write("[Request Error]\n")
                progress = 40 + int((i + 1) / total_chunks * 40)
                progress_callback(progress)

    print(f"Raw transcription saved as: {transcription_path}")

    # Step 4: Optionally reformat transcript using ChatGPT
    if organize_transcript:
        print("Formatting transcript using ChatGPT API...")
        with open(transcription_path, "r", encoding="utf-8") as raw_file:
            raw_transcript = raw_file.read()
        formatted_text = reformat_transcript(raw_transcript, progress_callback)
        with open(formatted_transcription_path, "w", encoding="utf-8") as formatted_file:
            formatted_file.write(formatted_text)
        print(f"Formatted transcription saved as: {formatted_transcription_path}")
    else:
        print("Skipping transcript formatting as per user selection.")

    # Step 5: Clean up chunk files
    print("Deleting audio chunks...")
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    print(f"Chunks deleted for {lecture_name}")
    progress_callback(100)


def browse_lectures():
    files = filedialog.askopenfilenames(
        title="Select Lecture Files",
        filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")]
    )
    # Clear previous file entries
    for widget in file_frame.winfo_children():
        widget.destroy()
    lecture_files.clear()
    progress_bars.clear()
    # Create a row with a label and progress bar for each file.
    for file in files:
        lecture_files.append(file)
        row = tk.Frame(file_frame)
        row.pack(fill=tk.X, pady=2)
        label = tk.Label(row, text=file, anchor="w")
        label.pack(side=tk.LEFT, padx=5)
        progress = ttk.Progressbar(row, orient="horizontal", length=200, mode="determinate", maximum=100)
        progress.pack(side=tk.RIGHT, padx=5)
        progress_bars[file] = progress


def start_processing():
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    if not output_dir:
        print("No output directory selected.")
        return
    model_choice = model_var.get()
    organize_transcript = organize_var.get()
    # If using Google API, get the selected language.
    google_language = "en-US"  # default
    if model_choice == "Google Web Speech API":
        lang_choice = google_lang_var.get()
        google_language = "pa-IN" if lang_choice == "Punjabi" else "en-US"

    # Process each file sequentially in a separate thread.
    def process_files():
        for file in lecture_files:
            progress_bar = progress_bars[file]

            def update_progress(val):
                progress_bar.config(value=val)
                root.update_idletasks()

            process_lecture(file, output_dir, model_choice, organize_transcript, google_language, update_progress)

    threading.Thread(target=process_files, daemon=True).start()


def update_language_frame(*args):
    # Show the language selection frame only if Google API is selected.
    if model_var.get() == "Google Web Speech API":
        google_lang_frame.pack(pady=5, padx=10, fill="x")
    else:
        google_lang_frame.forget()


# --------------------- GUI Setup ---------------------
root = tk.Tk()
root.title("Lecture Processor")

# Model Selection Frame
model_var = tk.StringVar(value="OpenAI Whisper")
model_frame = tk.LabelFrame(root, text="Select Transcription Model")
model_frame.pack(pady=10, padx=10, fill="x")
whisper_radio = tk.Radiobutton(model_frame, text="OpenAI Whisper", variable=model_var, value="OpenAI Whisper")
whisper_radio.pack(side=tk.LEFT, padx=10, pady=5)
google_radio = tk.Radiobutton(model_frame, text="Google Web Speech API", variable=model_var,
                              value="Google Web Speech API")
google_radio.pack(side=tk.LEFT, padx=10, pady=5)

# Google API Language Selection Frame (shown only when Google API is selected)
google_lang_var = tk.StringVar(value="English")
google_lang_frame = tk.LabelFrame(root, text="Select Language for Google API")
if model_var.get() == "Google Web Speech API":
    google_lang_frame.pack(pady=5, padx=10, fill="x")
model_var.trace("w", update_language_frame)

# Add radio buttons for language choice inside the google_lang_frame.
english_radio = tk.Radiobutton(google_lang_frame, text="English", variable=google_lang_var, value="English")
english_radio.pack(side=tk.LEFT, padx=10, pady=5)
punjabi_radio = tk.Radiobutton(google_lang_frame, text="Punjabi", variable=google_lang_var, value="Punjabi")
punjabi_radio.pack(side=tk.LEFT, padx=10, pady=5)

# Option to format transcript using ChatGPT.
organize_var = tk.BooleanVar(value=False)
organize_check = tk.Checkbutton(root, text="Organize Transcript (Format with open AI)", variable=organize_var)
organize_check.pack(pady=5)

# File List Frame (for file names and progress bars)
file_frame = tk.Frame(root)
file_frame.pack(pady=10, padx=10, fill="both", expand=True)

# Control Buttons Frame
button_frame = tk.Frame(root)
button_frame.pack(pady=10)
browse_button = tk.Button(button_frame, text="Browse Lectures", command=browse_lectures)
browse_button.pack(side=tk.LEFT, padx=10)
process_button = tk.Button(button_frame, text="Start Processing", command=start_processing)
process_button.pack(side=tk.LEFT, padx=10)
quit_button = tk.Button(button_frame, text="Quit", command=root.quit)
quit_button.pack(side=tk.LEFT, padx=10)

# Global lists/dictionaries to store file paths and their associated progress bars.
lecture_files = []
progress_bars = {}

root.mainloop()
