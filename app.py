from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import ffmpeg

import io
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for rendering

import matplotlib.pyplot as plt 
import numpy as np
import base64
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return "Voice backend is running"




from ffmpeg import Error as FFmpegError

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    try:
        webm_audio = file.read()
        print("\n---- New upload request ----")
        print(f"Time: {time.time()}")
        print(f"Uploaded audio size: {len(webm_audio)} bytes")

        # Safety check for bad recordings
        if len(webm_audio) < 1000:
            print(f"❌ Audio upload too small: {len(webm_audio)} bytes — skipping ffmpeg.")
            return jsonify({'error': f'Audio upload too small: {len(webm_audio)} bytes. Try again.'}), 400


        try:
            out, err = (
                ffmpeg
                .input('pipe:0')
                .output('pipe:1', format='wav', acodec='pcm_s16le', ac=1, ar='44100')
                .run(input=webm_audio, capture_stdout=True, capture_stderr=True)
            )
            print("FFMPEG stderr output:")
            print(err.decode())

        except FFmpegError as fferr:
            print("FFMPEG failed:")
            print(fferr.stderr.decode())
            return jsonify({'error': 'FFMPEG failed to process audio. Possibly invalid or corrupt input.'}), 400

        # Load WAV data
        wav_io = io.BytesIO(out)
        wav_io.seek(0)

        data, samplerate = sf.read(wav_io)
        num_samples = len(data)
        duration_sec = num_samples / samplerate

        if data.ndim > 1:
            data = data[:, 0]  # take one channel if stereo

        time_axis = np.linspace(0, duration_sec, len(data))

        # Plot waveform
        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time_axis, data, linewidth=1)
        ax.set_title("Time Domain Signal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        plt.grid()
        fig.tight_layout()

        svg_image = io.BytesIO()
        fig.savefig(svg_image, format='svg')
        plt.close(fig)
        svg_image.seek(0)
        plot_data = base64.b64encode(svg_image.getvalue()).decode('utf-8')

        # Pitch detection
        spectrum = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1 / samplerate)
        mask = (freqs > 100) & (freqs < 450)
        freqs_filtered = freqs[mask]
        spectrum_filtered = np.abs(spectrum[mask])

        if len(freqs_filtered) > 0:
            peak_index = np.argmax(spectrum_filtered)
            pitch_freq = freqs_filtered[peak_index]
            gender = "female" if pitch_freq > 175 else "male"
            pitch_message = f"The pitch frequency of your voice is {pitch_freq:.1f} Hz. You are most likely a {gender}."
        else:
            pitch_message = "Could not determine pitch frequency."

        return jsonify({
            'summary': pitch_message,
            'plot': plot_data,
            'format': 'svg'
        })

    except Exception as e:
        print("Unexpected exception occurred:", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
