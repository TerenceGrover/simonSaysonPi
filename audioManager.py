from pathlib import Path
import subprocess
import random
import os

AUDIO_ROOT = Path("audio")
SIMON_DIR = AUDIO_ROOT / "simon_says"
COMMANDS_DIR = AUDIO_ROOT / "commands"
SILENCE_WAV = AUDIO_ROOT / "silence_200ms.wav"

def play_wav(path):
    if not path or not os.path.exists(path):
        return

    # Wake Bluetooth first
    subprocess.run(
        ["aplay", str(SILENCE_WAV)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    subprocess.run(
        ["aplay", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def play_prompt_audio(target_name, simon):

    cmd_dir = COMMANDS_DIR / target_name
    if not cmd_dir.exists():
        print(f"[WARN] Missing command audio for {target_name}")
        return

    # Simon says (random variant)
    if simon:
        play_random_wav_from_folder(SIMON_DIR)
        play_random_wav_from_folder(cmd_dir)
    else:
        play_random_wav_from_folder(cmd_dir)

def play_random_wav_from_folder(folder: Path):
    if not folder.exists() or not folder.is_dir():
        print(f"[WARN] Missing audio folder: {folder}")
        return

    files = sorted(folder.glob("*.wav"))
    if not files:
        print(f"[WARN] No wav files in: {folder}")
        return

    wav = random.choice(files)
    play_wav(str(wav))