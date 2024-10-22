"""Live captions from microphone using Moonshine and SileroVAD models."""
import time

from queue import Queue

import keras
import numpy as np
import tokenizers

from silero_vad import load_silero_vad, VADIterator
from sounddevice import InputStream

from moonshine import load_model, ASSETS_DIR

SAMPLING_RATE = 16000

LOOKBACK_CHUNKS = 5
MARKER_LENGTH = 6
MAX_LINE_LENGTH = 80

MAX_SPEECH_DURATION = 15
REFRESH_SECS = 0.5   # Set higher with slower CPU or larger model size.

VERBOSE = False


class Transcriber(object):
    """Moonshine model transcriber."""
    def __init__(self):
        self.model = load_model("moonshine/tiny")
        tokenizer_file = ASSETS_DIR / "tokenizer.json"
        self.tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_file))
        self.__call__(np.zeros(int(SAMPLING_RATE)))  # Warmup.

    def __call__(self, speech):
        """Returns string containing Moonshine transcription of speech."""
        y = keras.ops.expand_dims(keras.ops.convert_to_tensor(speech), 0)
        tokens = self.model.generate(y)
        return self.tokenizer.decode_batch(tokens)[0]


def create_source_callback(q):
    def source_callback(data, frames, time, status):
        if status:
            print(status)
        q.put((data.copy(), status))
    return source_callback


def end_recording(speech, marker=""):
    """Transcribes, caches and prints the caption.  Zeroes speech."""
    if len(marker) != MARKER_LENGTH:
        raise ValueError("Unexpected marker length.")
    text = transcribe(speech)
    caption_cache.append(text + " " + marker)
    print_captions(text + (" " + marker) if VERBOSE else "")
    speech *= 0.0  # Clear speech buffer.


def print_captions(text):
    """Prints right justified on same line, prepending cached captions."""
    print('\r' + " " * MAX_LINE_LENGTH, end='', flush=True)
    if len(text) > MAX_LINE_LENGTH:
        text = text[-MAX_LINE_LENGTH:]
    elif text != "\n":
        for caption in caption_cache[::-1]:
            text = (caption[:-MARKER_LENGTH] if not VERBOSE else
                    caption + " ") + text
            if len(text) > MAX_LINE_LENGTH:
                break
        if len(text) > MAX_LINE_LENGTH:
            text = text[-MAX_LINE_LENGTH:]
    text = " " * (MAX_LINE_LENGTH - len(text)) + text
    print('\r' + text, end='', flush=True)


print("Loading Moonshine model ...")
transcribe = Transcriber()

vad_model = load_silero_vad(onnx=True)
vad_iterator = VADIterator(
    model=vad_model,
    sampling_rate=SAMPLING_RATE,
    threshold=0.5,
    min_silence_duration_ms=2000,
    speech_pad_ms=400,
)

q = Queue()
stream = InputStream(
    samplerate=SAMPLING_RATE,
    channels=1,
    blocksize=512 if SAMPLING_RATE == 16000 else 256,
    dtype=np.float32,
    callback=create_source_callback(q),
)
stream.start()

caption_cache = []
lookback_size = LOOKBACK_CHUNKS * 512 if SAMPLING_RATE == 16000 else 256
speech = np.empty(0, dtype=np.float32)

recording = False

print("Press Ctrl+C to quit live captions.\n")

with stream:
    print_captions("Ready...")
    try:
        while True:
            data, status = q.get()
            if VERBOSE and status:
                print(status)

            chunk = np.array(data).flatten()

            speech = np.concatenate((speech, chunk))
            if not recording:
                speech = speech[-lookback_size:]

            speech_dict = vad_iterator(chunk)
            if speech_dict:
                if 'start' in speech_dict and not recording:
                    recording = True
                    start_time = time.time()

                if 'end' in speech_dict and recording:
                    recording = False
                    end_recording(speech, "<STOP>")

            elif recording:
                # Possible speech truncation can cause hallucination.
                if (len(speech) / SAMPLING_RATE) > MAX_SPEECH_DURATION:
                    recording = False
                    end_recording(speech, "<SNIP>")
                    # Soft reset without affecting VAD model state.
                    vad_iterator.triggered = False
                    vad_iterator.temp_end = 0
                    vad_iterator.current_sample = 0

                if (time.time() - start_time) > REFRESH_SECS:
                    print_captions(transcribe(speech))
                    start_time = time.time()

    except KeyboardInterrupt:
        stream.close()

        if caption_cache:
            print("\nCached captions.")
            for caption in caption_cache:
                print(caption[:-MARKER_LENGTH], end="", flush=True)
        print("")
