# Citadel

This project implements Citadel.

## Prerequisites

- Python 3.9.0
- pip (Python package installer)

## Installation

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd NER
   ```

2. Create and activate a virtual environment
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install required packages
   ```bash
   pip install -r requirements.txt
   ```

4. Download SpaCy models. Turkish model is not available on SpaCy. You can download it locally.
   ```bash
   python -m spacy download en_core_web_trf
   pip install tr_core_news_trf-3.4.2-py3-none-any.whl
   ```