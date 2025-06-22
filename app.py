from flask import Flask, render_template, request, flash, jsonify
from werkzeug.utils import secure_filename
# from predictSpeech import extract_audio_features, extract_text_features, transcribe_audio
import tempfile
import os
import io


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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/speechAnalysis", methods=["GET", "POST"])
def speechAnalysis():
    if request.method == "GET":
        return render_template("speechAnalysis.html", context=projects["speechAnalysis"])
    elif request.method == "POST":
        if 'audio-file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['audio-file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # Secure the filename to prevent path traversal
            filename = secure_filename(file.filename)

            # Read the file into memory (as a BytesIO object)
            file_content = io.BytesIO(file.read())
        
            try:
                # Use tempfile to create a temporary file-like object from the BytesIO object
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file_content.getvalue())
                    temp_file_path = temp_file.name
                    text = transcribe_audio(temp_file_path)
                    os.remove(temp_file_path)
                    response = {
                        'message': 'Audio analysis completed successfully.',
                        'filename': filename,
                        'transcription': text  # Example result: transcribed text
                    }

                    return jsonify(response), 200
            except Exception as e:
                return jsonify({'error': f'Error processing audio file: {str(e)}'}), 500
            model_path = "static/assets/models/audio_only_reg_model_t70.pth"
            DEVICE = "cuda"
            response = {
                'message': 'Audio analysis completed successfully.',
                'filename': filename,
                'analysis': 'Mock analysis result for audio file.'
            }
            return jsonify(response), 200

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)

