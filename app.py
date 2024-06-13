from flask import Flask, request, jsonify, render_template, send_file, url_for, redirect
import os
import threading
import cv2
import subprocess
import time
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO("best.pt")

# Shared state for tracking progress
progress = {
    'status': 'idle',
    'progress': 0,
    'filename': None,
    'avg_inference_time': 0
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global progress
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        output_path = os.path.join(RESULT_FOLDER, os.path.splitext(file.filename)[0] + '.mp4' if file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) else file.filename)
        
        # Start a background thread to process the file
        threading.Thread(target=process_file, args=(filename, output_path)).start()

        progress['status'] = 'processing'
        progress['progress'] = 0
        progress['filename'] = file.filename

        return render_template('progress.html')

@app.route('/progress')
def get_progress():
    return jsonify(progress)

def process_file(input_path, output_path):
    global progress
    progress['status'] = 'processing'
    
    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        inference_times = process_video(input_path, output_path)
    else:
        image = cv2.imread(input_path)
        start_time = time.time()
        results = model(image)
        end_time = time.time()
        annotated_image = results[0].plot()
        cv2.imwrite(output_path, annotated_image)
        inference_times = [end_time - start_time]
        
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    avg_inference_time = round(avg_inference_time, 3)  # Round to three decimal places

    progress['status'] = 'completed'
    progress['progress'] = 100
    progress['avg_inference_time'] = avg_inference_time

def process_video(input_path, output_path):
    global progress
    temp_output = "temp_" + os.path.basename(output_path)
    video = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, 20.0, (int(video.get(3)), int(video.get(4))))
    
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    inference_times = []

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        start_time = time.time()
        results = model(frame)
        end_time = time.time()

        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        processed_frames += 1
        inference_times.append(end_time - start_time)
        progress['progress'] = int((processed_frames / frame_count) * 100)

    video.release()
    out.release()

    # Use FFmpeg to encode the video properly
    ffmpeg_command = f"ffmpeg -y -i {temp_output} -c:v libx264 -preset slow -crf 22 -c:a copy {output_path}"
    subprocess.call(ffmpeg_command, shell=True)
    os.remove(temp_output)
    
    return inference_times

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

@app.route('/result/<filename>')
def result_file(filename):
    avg_inference_time = progress.get('avg_inference_time', 0)
    return render_template('result.html', filename=filename, avg_inference_time=avg_inference_time)

if __name__ == '__main__':
    app.run(debug=True)
