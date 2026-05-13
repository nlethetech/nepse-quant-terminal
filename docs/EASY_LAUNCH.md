# Easy Launch

NEPSE Quant Terminal can be launched without manually creating a virtual environment.

## macOS

Double-click:

- `Nepse Quant Terminal.app`, or
- `Launch Quant Terminal.command`

The launcher opens Terminal, creates `.venv` if needed, installs `requirements.txt`, downloads the bundled market database if missing, runs preflight, then starts the TUI.

Keep `Nepse Quant Terminal.app` inside the repo folder. It is a lightweight launcher that uses the files beside it; do not drag it into `/Applications` by itself.

If macOS blocks the app because it was downloaded from the internet:

1. Right-click `Nepse Quant Terminal.app`
2. Click **Open**
3. Confirm **Open**

Python 3.12 is recommended. Python 3.10 through 3.13 is supported.

## Windows

Double-click:

- `Launch Quant Terminal.bat`

The launcher creates `.venv`, installs dependencies, downloads the bundled market database if missing, runs preflight, then starts the TUI.

Python 3.12 is recommended. During Python install, select **Add python.exe to PATH**.

## What Gets Created

The launcher writes only local runtime files:

- `.venv/`
- `data/nepse_market_data.db` if missing
- `data/runtime/`

These are local machine files and should not be committed.

## Troubleshooting

- If dependency installation fails, install Git and launch again. The NEPSE package is installed from GitHub.
- If the dashboard opens but data is stale, run `python setup_data.py --scrape --days 90` from the activated environment or use the normal scraper workflow.
- If the terminal closes immediately, run the `.command` or `.bat` launcher directly so the error remains visible.
