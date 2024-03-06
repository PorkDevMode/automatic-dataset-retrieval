import os
from pydub import AudioSegment, silence
import assemblyai as aai
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import subprocess
import uuid as uid
import noisereduce as nr
import librosa
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='\033[1;34m%(asctime)s\033[1;0m - \033[1;32m%(levelname)s\033[1;0m - \033[1;33m%(message)s\033[1;0m')


random_dir = str(uid.uuid4())

with open('aai_key.txt', 'r') as file:
    api_key = file.readline().strip()

if not api_key:
    print("Error: The API key is blank. Please provide a correct AssemblyAI API key in aai_key.txt. Please read the readme to learn how.")
else:
    aai.settings.api_key = api_key

aai.settings.api_key = api_key

SCOPES = ['https://www.googleapis.com/auth/drive.file']
script_dir = os.path.dirname(os.path.abspath(__file__))
unfiltered_dir = os.path.join(script_dir, 'unfiltered')
output_base_dir = os.path.join(script_dir, 'filtered', random_dir)
final_audio_path = os.path.join(script_dir, 'final_audio.mp3')

os.makedirs(unfiltered_dir, exist_ok=True)
os.makedirs(output_base_dir, exist_ok=True)

def convert_and_combine_mp4_to_mp3(unfiltered_dir, combined_mp3_path):
    combined_audio = AudioSegment.silent(duration=0)
    for filename in os.listdir(unfiltered_dir):
        if filename.endswith(".mp4"):
            source_path = os.path.join(unfiltered_dir, filename)
            audio = AudioSegment.from_file(source_path, "mp4")
            combined_audio += audio

    final_audio = combined_audio.set_channels(1).set_frame_rate(22050).set_sample_width(2)

    final_audio.export(combined_mp3_path, format="mp3")

def remove_silence(audio_segment, silence_thresh=-50, min_silence_len=2000):
    non_silent_audio = silence.split_on_silence(audio_segment,
                                                 min_silence_len=min_silence_len,
                                                 silence_thresh=silence_thresh)
    return sum(non_silent_audio, AudioSegment.silent(duration=0))

def separate_vocals_with_demucs(source_path, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    cmd = [
        'python', '-m', 'demucs', source_path, target_dir
    ]
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Demucs Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Demucs encountered an error:")
        print(e.stderr)

def process_audio_files(unfiltered_dir, final_audio_path):
    logging.info("Starting audio file processing.")
    temp_dir = os.path.join(unfiltered_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    combined_mp3_path = os.path.join(temp_dir, "combined.mp3")
    logging.info(f"Combining MP4 files in {unfiltered_dir} into {combined_mp3_path}.")
    convert_and_combine_mp4_to_mp3(unfiltered_dir, combined_mp3_path)
    
    logging.info(f"Separating vocals using Demucs from {combined_mp3_path}.")
    separate_vocals_with_demucs(combined_mp3_path, temp_dir)
    
    vocal_track_path = os.path.join(script_dir, 'separated', 'htdemucs', 'combined', 'vocals.wav')
    logging.info(f"Removing silence from the vocal track at {vocal_track_path}.")
    vocal_audio = AudioSegment.from_wav(vocal_track_path)
    processed_audio = remove_silence(vocal_audio)
    processed_audio.export(final_audio_path, format="mp3")
    logging.info(f"Exported final audio to {final_audio_path}.")

def upload_to_drive(file_path):
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('config.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('drive', 'v3', credentials=creds)
    file_metadata = {'name': os.path.basename(file_path)}
    media = MediaFileUpload(file_path, mimetype='audio/mpeg')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"File ID: {file.get('id')}")
    service.permissions().create(fileId=file.get('id'), body={"type": "anyone", "role": "reader"}).execute()
    return f"https://drive.google.com/uc?id={file.get('id')}"

def diarize_audio(audio_url):
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcript = aai.Transcriber().transcribe(data=audio_url, config=config)
    if transcript.status == 'completed':
        return transcript
    else:
        raise Exception(f'Transcription failed with status: {transcript.status}')

def process_diarization_results(transcript, audio_segment):
    if hasattr(transcript, 'utterances'):
        speaker_data = {}
        for utterance in transcript.utterances:
            speaker = utterance.speaker
            if speaker not in speaker_data:
                speaker_data[speaker] = []
            speaker_data[speaker].append({
                'start': utterance.start,
                'end': utterance.end
            })
        save_speaker_audio(speaker_data, audio_segment)
    else:
        print("No utterances found in transcript.")

def save_speaker_audio(speaker_data, audio_segment):
    for speaker, phrases in speaker_data.items():
        speaker_dir = os.path.join(output_base_dir, f"speaker_{speaker}")
        os.makedirs(speaker_dir, exist_ok=True)
        for idx, phrase in enumerate(phrases):
            speaker_audio = audio_segment[phrase['start']:phrase['end']]
            speaker_audio.export(os.path.join(speaker_dir, f"{idx}.wav"), format="wav")
            
print("\033[1;36mStep 0: Processing Audio Files\033[1;0m")
process_audio_files(unfiltered_dir, final_audio_path)
print("\033[1;36mStep 1: Uploading to Drive\033[1;0m")
audio_url = upload_to_drive(final_audio_path)
print("\033[1;36mStep 2: Diarizing Audio\033[1;0m")
results = diarize_audio(audio_url)
print("\033[1;36mStep 3: Processing Diarization Results\033[1;0m")
audio_segment = AudioSegment.from_file(final_audio_path)
process_diarization_results(results, audio_segment)
print("\033[1;36mStep 4: Completed\033[1;0m")