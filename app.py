from flask import Flask, render_template, request, flash, jsonify
from werkzeug.utils import secure_filename
from predictSpeech import extract_audio_features, extract_text_features, transcribe_audio
import tempfile
import os
import io
from pydub import AudioSegment 
from evaluator_class import get_preds_class
from evaluator_reg import get_preds_reg

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'ogg'}

@app.route('/', methods=["GET"])
def index():
    if request.method == "GET":
        return render_template('index.html')

@app.route("/projects", methods=["GET"])
def projects():
    if request.method == "GET":
        return render_template('projects.html')

projects = {
    "speechAnalysis": {
        "title": "Is an Influencer’s Speech a Good Predictor of their Popularity in Media?",
        "description": "__",
        "ghlink": "--"
    },
    "2": {
        "title": "--",
        "description": "__",
        "ghlink": "--"
    }
}

# Replace this with your actual logic
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp3', 'wav', 'm4a', 'ogg'}

@app.route("/speechAnalysis", methods=["GET", "POST"])
def speechAnalysis():
    if request.method == "GET":
        return render_template("speechAnalysis.html", context={"title": "Speech Analysis"})

    elif request.method == "POST":
        print("recieved")
        if 'audio-file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['audio-file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_ext = filename.rsplit('.', 1)[1].lower()

            try:
                # Read uploaded file into memory
                audio_data = io.BytesIO(file.read())
                # Convert audio to WAV if needed
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                    if file_ext != "wav":
                        audio = AudioSegment.from_file(audio_data, format=file_ext)
                        audio.export(temp_wav.name, format="wav")
                    else:
                        temp_wav.write(audio_data.getvalue())
                    wav_path = temp_wav.name
                print(request.form.get("model"))
                if request.form.get("model") == "rmultimodal":
                    text_transcribed = transcribe_audio(wav_path)
                    audio = extract_audio_features(wav_path)
                    text = extract_text_features(text_transcribed)
                    results = get_preds_reg("both",text,audio)
                    print(results)
                elif request.form.get("model") == "cmultimodal":
                    text_transcribed = transcribe_audio(wav_path)
                    text = extract_text_features(text_transcribed)
                    audio = extract_audio_features(wav_path)
                    results = get_preds_class("both",text,audio)
                    print(results)
                elif request.form.get("model") == "raudio":
                    audio = extract_audio_features(wav_path)
                    results = get_preds_reg("audio",audio=audio)
                    print(results)
                elif request.form.get("model") == "caudio":
                    audio = extract_audio_features(wav_path)
                    results = get_preds_class("audio",audio=audio)
                    print(results)
                elif request.form.get("model") == "rtext":
                    text_transcribed = transcribe_audio(wav_path)
                    text = extract_text_features(text_transcribed)
                    results = get_preds_reg("text",text=text)
                    print(results)
                elif request.form.get("model") == "ctext":
                    text_transcribed = transcribe_audio(wav_path)
                    text = extract_text_features(text_transcribed)
                    results = get_preds_class("text",text=text)
                    print(results)
                # Clean up temp file
                os.remove(wav_path)

                return jsonify({
                    'message': 'Audio analysis completed successfully.',
                    'results': results,
                    'mode': request.form.get("model")
                }), 200

            except Exception as e:
                return jsonify({'error': f'Error processing audio file: {str(e)}'}), 500

        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)

