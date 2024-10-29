"""Live captions from microphone using Moonshine and SileroVAD ONNX models."""

import argparse
import os
import sys
import time

from queue import Queue

import numpy as np

from silero_vad import load_silero_vad, VADIterator
from sounddevice import InputStream
from tokenizers import Tokenizer

# Local import of Moonshine ONNX model.
MOONSHINE_DEMO_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(MOONSHINE_DEMO_DIR, ".."))

from onnx_model import MoonshineOnnxModel

SAMPLING_RATE = 16000

CHUNK_SIZE = 512  # Silero VAD requirement with sampling rate 16000.
LOOKBACK_CHUNKS = 5
MARKER_LENGTH = 6
MAX_LINE_LENGTH = 80

# These affect live caption updating - adjust for your platform speed and model.
MAX_SPEECH_SECS = 15
MIN_REFRESH_SECS = 0.2

VERBOSE = False


class Transcriber(object):
    def __init__(self, model_name, rate=16000):
        if rate != 16000:
            raise ValueError("Moonshine supports sampling rate 16000 Hz.")
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.rate = rate
        assets_dir = f"{os.path.join(os.path.dirname(__file__), '..', 'assets')}"
        tokenizer_file = f"{assets_dir}{os.sep}tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))

        self.inference_secs = 0
        self.number_inferences = 0
        self.speech_secs = 0
        self.__call__(np.zeros(int(rate), dtype=np.float32))  # Warmup.

    def __call__(self, speech):
        """Returns string containing Moonshine transcription of speech."""
        self.number_inferences += 1
        self.speech_secs += len(speech) / self.rate
        start_time = time.time()

        tokens = self.model.generate(speech[np.newaxis, :].astype(np.float32))
        text = self.tokenizer.decode_batch(tokens)[0]

        self.inference_secs += time.time() - start_time
        return text


def create_input_callback(q):
    """Callback method for sounddevice InputStream."""

    def input_callback(data, frames, time, status):
        if status:
            print(status)
        q.put((data.copy().flatten(), status))

    return input_callback


def end_recording(speech, marker=""):
    """Transcribes, caches and prints the caption.  Clears speech buffer."""
    if len(marker) != MARKER_LENGTH:
        raise ValueError("Unexpected marker length.")
    text = transcribe(speech)
    caption_cache.append(text + " " + marker)
    print_captions(text + (" " + marker) if VERBOSE else "", True)
    speech *= 0.0


def print_captions(text, new_cached_caption=False):
    """Prints right justified on same line, prepending cached captions."""
    print("\r" + " " * MAX_LINE_LENGTH, end="", flush=True)
    if len(text) > MAX_LINE_LENGTH:
        text = text[-MAX_LINE_LENGTH:]
    elif text != "\n":
        for caption in caption_cache[::-1]:
            text = (caption[:-MARKER_LENGTH] if not VERBOSE else caption + " ") + text
            if len(text) > MAX_LINE_LENGTH:
                break
        if len(text) > MAX_LINE_LENGTH:
            text = text[-MAX_LINE_LENGTH:]
    text = " " * (MAX_LINE_LENGTH - len(text)) + text
    print("\r" + text, end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="live_captions",
        description="Live captioning demo of Moonshine models",
    )
    parser.add_argument(
        "--model_name",
        help="Model to run the demo with",
        default="moonshine/base",
        choices=["moonshine/base", "moonshine/tiny"],
    )
    args = parser.parse_args()
    model_name = args.model_name
    print(f"Loading Moonshine model '{model_name}' (using ONNX runtime) ...")
    transcribe = Transcriber(model_name=model_name, rate=SAMPLING_RATE)

    vad_model = load_silero_vad(onnx=True)
    vad_iterator = VADIterator(
        model=vad_model,
        sampling_rate=SAMPLING_RATE,
        threshold=0.5,
        min_silence_duration_ms=300,
    )

    q = Queue()
    stream = InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,
        blocksize=CHUNK_SIZE,
        dtype=np.float32,
        callback=create_input_callback(q),
    )
    stream.start()

    caption_cache = []
    lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE
    speech = np.empty(0, dtype=np.float32)

    recording = False

    print("Press Ctrl+C to quit live captions.\n")

    with stream:
        print_captions("Ready...")
        try:
            while True:
                chunk, status = q.get()
                if VERBOSE and status:
                    print(status)

                speech = np.concatenate((speech, chunk))
                if not recording:
                    speech = speech[-lookback_size:]

                speech_dict = vad_iterator(chunk)
                if speech_dict:
                    if "start" in speech_dict and not recording:
                        recording = True
                        start_time = time.time()

                    if "end" in speech_dict and recording:
                        recording = False
                        end_recording(speech, "<STOP>")

                elif recording:
                    # Possible speech truncation can cause hallucination.

                    if (len(speech) / SAMPLING_RATE) > MAX_SPEECH_SECS:
                        recording = False
                        end_recording(speech, "<SNIP>")
                        # Soft reset without affecting VAD model state.
                        vad_iterator.triggered = False
                        vad_iterator.temp_end = 0
                        vad_iterator.current_sample = 0

                    if (time.time() - start_time) > MIN_REFRESH_SECS:
                        print_captions(transcribe(speech))
                        start_time = time.time()

        except KeyboardInterrupt:
            stream.close()

            if recording:
                while not q.empty():
                    chunk, _ = q.get()
                    speech = np.concatenate((speech, chunk))
                end_recording(speech, "<END.>")

            print(f"""

             model_name :  {model_name}
       MIN_REFRESH_SECS :  {MIN_REFRESH_SECS}s

      number inferences :  {transcribe.number_inferences}
    mean inference time :  {(transcribe.inference_secs / transcribe.number_inferences):.2f}s
  model realtime factor :  {(transcribe.speech_secs / transcribe.inference_secs):0.2f}x
""")
            if caption_cache:
                print("Cached captions.")
                for caption in caption_cache:
                    print(caption[:-MARKER_LENGTH], end="", flush=True)
                print("")
