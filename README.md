# Spinal AI Project

## Overview
The Spinal AI Project is a web application designed for the diagnosis of cervical disk herniation using MRI images. The application leverages advanced image processing techniques to analyze MRI scans and generate diagnostic reports.

## Project Structure
```
spinal-ai-project
├── backend.py       # Core functionality for image processing and analysis
├── app.py           # Flask web application for handling file uploads and processing
├── index.html       # Front-end interface for user interaction
└── README.md        # Documentation for the project
```

## File Descriptions

### backend.py
This file contains the core functionality of the application, including:
- Image processing and analysis for cervical disk herniation diagnosis.
- Functions for detecting and classifying MRI images.
- Cropping images and generating diagnosis reports.

**Key Functions:**
- `get_perfect_slices`: Analyzes MRI images and retrieves slices based on detection.
- `plot_bboxes`: Draws bounding boxes around detected areas in the MRI images.
- `generate_report`: Creates a diagnostic report based on the analysis results.

### app.py
This file sets up a Flask web application that:
- Handles file uploads for MRI images.
- Retrieves patient details from the user.
- Calls functions from `backend.py` to analyze the images and generate reports.
- Manages the response format, including returning images in base64 format.

### index.html
This file serves as the front-end interface for the application, featuring:
- A form for users to input patient details and upload MRI images.
- Styles for layout and design to enhance user experience.
- JavaScript for handling form submission, displaying results, and managing image interactions.

## Setup Instructions
1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd spinal-ai-project
   ```

2. **Install dependencies:**
   Ensure you have Python and pip installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application:**
   Start the Flask application by executing:
   ```
   python app.py
   ```

4. **Access the application:**
   Open your web browser and navigate to `http://127.0.0.1:5000` to access the interface.

## Usage Guidelines
- Fill in the patient details in the provided form.
- Upload the MRI images (only axial C2-T1 MRI images are allowed).
- Submit the form to receive a diagnosis report along with the processed images.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.
