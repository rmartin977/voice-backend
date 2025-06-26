from flask import Flask, request, jsonify, render_template
import ffmpeg

import io
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for rendering

import matplotlib.pyplot as plt
import numpy as np
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    try:
        webm_audio = file.read()

# Use ffmpeg-python to convert .webm to .wav
        process = (
            ffmpeg
            .input('pipe:0')
            .output('pipe:1', format='wav', acodec='pcm_s16le', ac=1, ar='44100')
            .run(input=webm_audio, capture_stdout=True, capture_stderr=True)
        )

        wav_bytes = process[0]
        wav_io = io.BytesIO(wav_bytes)
        wav_io.seek(0)


        data, samplerate = sf.read(wav_io)
        num_samples = len(data)
        duration_sec = num_samples / samplerate

        # If stereo, take one channel
        if data.ndim > 1:
            data = data[:, 0]

        time = np.linspace(0, duration_sec, len(data))

        # Create a plot
        fig = plt.figure(figsize=(8,3))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(time, data, linewidth=1)
        ax.set_title("Time Domain Signal")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()

        plt.grid()
        # Save plot to SVG in memory
        svg_image = io.BytesIO()
        fig.savefig(svg_image, format='svg')
        plt.close(fig)
        svg_image.seek(0)
        plot_data = base64.b64encode(svg_image.getvalue()).decode('utf-8')

        # --- Pitch Detection ---
        spectrum = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1 / samplerate)

        # Consider only the positive frequencies in human voice range
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

        message = f"Your message was recorded at {samplerate:,} samples/sec. " \
                  f"It contains {num_samples:,} samples and is {duration_sec:,.2f} seconds long.\n" + pitch_message

        return jsonify({
            'summary': pitch_message,
            'plot': plot_data,
            'format': 'svg'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
