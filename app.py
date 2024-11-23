import os
from flask import Flask, render_template, request
from transformers import pipeline
from werkzeug.utils import secure_filename
import PyPDF2

# Initialize Flask app
app = Flask(__name__)

# Directory to save uploaded files
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Initialize the text-classification pipeline with a valid model
model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def allowed_file(filename):
    """Check if the uploaded file is a PDF."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text content from a PDF file."""
    pdf_text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                pdf_text += page.extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return pdf_text

def analyze_resume(resume_text):
    """Analyze the extracted resume text and return ATS insights."""
    # Run the model on the resume text
    predictions = model(resume_text)

    # Example ATS analysis
    ats_score = calculate_ats_score(resume_text)
    feedback = "Include more keywords related to Python, Data Science, and Machine Learning."

    return {"score": ats_score, "feedback": feedback, "model_predictions": predictions}

def calculate_ats_score(resume_text):
    """Calculate a mock ATS score based on keyword matching."""
    keywords = ['python', 'data analysis', 'machine learning', 'SQL', 'cloud computing']
    matched = [kw for kw in keywords if kw.lower() in resume_text.lower()]
    score = len(matched) / len(keywords) * 100
    return round(score, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", result={"error": "No file part"})
        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", result={"error": "No selected file"})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Extract text from PDF
            resume_text = extract_text_from_pdf(file_path)
            if not resume_text.strip():
                return render_template("index.html", result={"error": "Failed to extract text from the PDF."})

            # Analyze the extracted text
            result = analyze_resume(resume_text)
            return render_template("index.html", result=result)
    return render_template("index.html", result=None)

if __name__ == "__main__":
    # Ensure the upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
