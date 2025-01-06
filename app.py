from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
from werkzeug.utils import secure_filename
from backend import get_perfect_slices , generate_report , plot_bboxes
import shutil
from ultralytics import RTDETR
import os
#pip install -r requirements.txt

app = Flask(__name__)
CORS(app)


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'mriImage' not in request.files:
            return jsonify({"error": "No file part"}), 400

        # Get patient details from the form data
        patient_id = request.form['patientId']
        dob = request.form['dob']
        gender = request.form['gender']
        age = request.form['age']

        # Get the list of MRI images
        mri_images = request.files.getlist('mriImage')

        results = []
        model = RTDETR("model_detection.pt")

        # Ensure the directory exists
        upload_folder = 'uploads'
        os.makedirs(upload_folder, exist_ok=True)

        # Process each MRI image
        for mri_image in mri_images:
            # Secure the filename and save the image to the server
            filename = secure_filename(mri_image.filename)
            file_path = os.path.join(upload_folder, filename)
            mri_image.save(file_path)
       
        images_disk_dic  = get_perfect_slices(upload_folder)
        final_images = []
        for i in images_disk_dic.keys():
            final_images.append(images_disk_dic[i])

        for mri_image in final_images:
            # Read the image file
            image = Image.open(mri_image)
            
            image = np.array(image)

            # Process the image using OpenCV (add text for demonstration)
            p = model.predict(image)
            res = plot_bboxes(p)


            diagnosis = res[2]
            level = res[1]

            # Encode image to base64
            _, buffer = cv2.imencode('.png', res[0])
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Store result for this image
            results.append({
                'patientId': patient_id,
                'age': age,
                'gender': gender,
                'diagnosis': diagnosis,
                'level': level,
                'image': image_base64
            })

        levels = []
        diag = []
        for i in results:
            levels.append(i['level'])
            diag.append(i['diagnosis'])

        report = generate_report(patient_id, age, gender, levels, diag)


        shutil.rmtree(upload_folder)
        return jsonify({
            'images': results , 'report':report
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
