# scientikz

Natural-language to TikZ/LaTeX generator for Math, Physics, Statistics, and Chemistry.  
Tech stack: Streamlit + Supabase (same pattern as Math2TikZ).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Fill in SUPABASE_URL and SUPABASE_KEY
streamlit run app.py
