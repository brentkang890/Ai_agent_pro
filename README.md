Pro Trader AI - Full Package (fixed)
Files included:
- main_combined.py (full implementation)
- requirements.txt
- Procfile
- .well-known/ai-plugin.json
- openapi.json
- README.md

Deployment (Railway):
1. Create GitHub repo and upload these files.
2. Connect repo to Railway and deploy. Set BACKTEST_URL env variable to your backtester endpoint (optional).
3. Test /health endpoint after deploy.

Notes:
- For chart OCR, server needs tesseract binary installed. Railway may not provide it.
- For best results, test analyze endpoints with CSV and clear screenshots.
