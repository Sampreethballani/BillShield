# BillShield

Turn a bill you don't trust into one message worth sending. BillShield reads a bill, checks its arithmetic in Python, compares its printed total with the amount you expected, and drafts a complaint message for you to review.

## Run locally on Windows

1. Install Python 3.12 or later and Tesseract OCR. If Tesseract is installed with winget, run `winget install --exact --id UB-Mannheim.TesseractOCR`, then reopen PowerShell.
2. In this folder, copy `.env.example` to `.env` and put your Groq API key in `.env`.
3. Install dependencies and start BillShield:

   ```powershell
   python -m pip install -r requirements.txt
   python app.py
   ```

4. Open the local URL printed in the terminal. Keep that terminal open while using the app.

The API key stays on the server. The local SQLAlchemy database is `.billshield_sqlalchemy.sqlite3`; use **Saved records** in the app to inspect saved bill, line-item, promise, and message audit rows. Bill files and message text are not stored in the database.

## Before deploying

The local database file is ignored by Git and must not be used as persistent storage on a serverless host. Configure a persistent hosted database and deployment entry point before deploying this app to Vercel.
