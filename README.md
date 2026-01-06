# 🛡️ Phishnix - AI-Based Phishing Detection System

![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Supervised-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Security](https://img.shields.io/badge/Security-High-brightgreen.svg)

Phishnix is a machine learning–powered phishing detection system designed to identify malicious URLs and protect users from online scams.  
This project improves cybersecurity awareness using real-world datasets and ML techniques.

---

## 🌟 Key Features

- **Phishing Detection**: Detect malicious URLs using supervised ML algorithms
- **Real Datasets**: Trained on legitimate and phishing URLs
- **Web Interface**: Flask-based responsive frontend
- **Fast Predictions**: ML model loaded via `pickle` for instant detection
- **Feature Extraction**: URL lexical and structural feature analysis
- **Secure & Robust**: Handles various URL formats with input validation

---

## 📋 Prerequisites

- **Runtime Environment**: Python 3.8+ (tested up to 3.11)  
- **Web Framework**: Flask 2.0+  
- **Dependencies**:
  - flask
  - scikit-learn
  - pandas
  - numpy
  - pickle
- **Datasets**: phishing.csv, legitimateurls.csv, phishurls.csv

---

## 🚀 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/iAryanbajaj/Phishnix1.2.git
cd Phishnix1.2
2. Install Dependencies
pip install -r requirements.txt
3. Run the Application
python app.py
4. Open in Browser
http://127.0.0.1:5000
🔒 Security & Validation
Input Validation
Ensures data integrity and prevents errors:
# Example URL input validation
if not url:
    flash('URL cannot be empty', 'danger')
    return redirect(url_for('index'))
ML Model Security
Model loaded via pickle for safe and fast predictions
Sanitized user input to prevent injection or invalid URL issues
🧠 Machine Learning Overview
Feature Extraction: Extracts lexical & structural URL features
Training: Supervised classification (RandomForest / SVM / Logistic Regression)
Prediction: Instant classification as phishing or legitimate
Persistence: Model saved as newmodel.pkl for production use
# Example: Loading model
import pickle
model = pickle.load(open('newmodel.pkl', 'rb'))
prediction = model.predict([url_features])
📁 Project Structure
Phishnix/
│── app.py
│── train_model.py
│── feature.py
│── convert.py
│── newmodel.pkl
│── requirements.txt
│── templates/
│   ├── index.html
│   └── usecases.html
│── static/
│── DataFiles/
│   ├── phishing.csv
│   ├── legitimateurls.csv
│   └── phishurls.csv
👥 Contributors
Aryan Bajaj – Project Lead, Backend & ML Development
GitHub: https://github.com/iAryanbajaj
[Friend Name] – Frontend / Data Analysis / Research
GitHub: https://github.com/FRIEND_GITHUB
Replace [Friend Name] and GitHub link with actual details.
📌 Future Enhancements
Real-time URL scanning
Browser extension for phishing detection
Cloud API deployment
Improved ML model accuracy
📄 License
This project is licensed under the MIT License - see LICENSE
🙏 Acknowledgments
Flask - Web framework
Scikit-learn - Machine learning library
Pandas & NumPy - Data manipulation
Real-world phishing datasets for training and testing
✅ Git Commands After Adding README
git add README.md
git commit -m "Updated README with contributors"
git push
