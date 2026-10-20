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

The app exports a FastAPI `app` from the repository-root `app.py`, which Vercel can detect automatically. For deployment:

1. Import the `Sampreethballani/BillShield` GitHub repository into Vercel.
2. Add `GROQ_API_KEY` as a Vercel environment variable. Do not commit `.env`.
3. Connect a hosted PostgreSQL database and set `DATABASE_URL` for the Vercel project. Without it, BillShield falls back to a local SQLite file, which is not durable on Vercel.

Vercel's standard Python runtime does not install the system Tesseract executable. Text-based PDFs can be processed without OCR, but scanned PDFs and images need a separately configured OCR provider before those uploads will work in the deployed app. Local Windows development still uses Tesseract as described above.
