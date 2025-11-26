WPP Demo Project
Analytics Dashboard + Recommender System Demo

(Created for internal review)

Welcome to the WPP Demo Project.
This repository contains two applications:

EDA App
A Streamlit dashboard for exploratory data analysis.

Recommender Dashboard
An interactive dashboard demonstrating the recommendation engine.

Both apps are fully self-contained and ready to run.

⭐ 1. Requirements

Please make sure you have the following installed:

✔️ Python 3.10 or newer

Check with:

python3 --version

✔️ pip

Check with:

pip --version

✔️ Git LFS

This is needed to download large .parquet files.

If you installed Git recently, GitHub handles LFS automatically.

⭐ 2. Cloning the Repository

If you have been added as a collaborator, you can clone using either HTTPS (easiest) or SSH (advanced).
Both methods are explained at the end of this document.

For most users, use HTTPS:

git clone https://github.com/farzadismyid/demo_wpp.git
cd demo_wpp

⭐ 3. Create and Activate a Virtual Environment

This keeps the project isolated from your system Python.

Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

Windows (PowerShell):
python -m venv .venv
.venv\Scripts\activate


When activated, your terminal will show:

(.venv) user@computer:~/demo_wpp$

⭐ 4. Install Dependencies

Run:

pip install -r requirements.txt


This installs:

Streamlit

Pandas

DuckDB

Numpy

All internal requirements

And automatically pulls the large .parquet files via Git LFS

⭐ 5. Running the Applications

There are two apps. Run them separately.

🔷 A) EDA App (Exploratory Data Analysis)

From the root folder demo_wpp run:

streamlit run eda_app.py


A browser window will open automatically showing:

Usage trends

Top categories

Merchant behavior

Time-based charts

Interactive filters

🔷 B) Recommender Dashboard

This dashboard must be started from the app/ folder because its data files are located there.

Steps:
cd app
streamlit run dashboard.py


This dashboard includes:

Global evaluation metrics

Precision@K, MAP@K, NDCG@K

Per-user recommendations

Merchant-level details

Category drill-downs

Session-level analysis

Filtering by merchant, user, or category

⭐ 6. Project Structure
demo_wpp/
│
├── eda_app.py                         # EDA dashboard
├── requirements.txt                   # Dependencies
├── wpp_2021_2023_final.parquet        # Data (Git LFS)
│
├── app/
│   ├── dashboard.py                   # Recommendation dashboard
│   ├── wpp_2020_2024_final.parquet    # Data (Git LFS)
│   ├── *.csv                          # Extra small data files
│   └── ...
│
└── engine/
    └── Backend engine code

⭐ 7. Updating the Project

To get the latest version:

git pull


If new packages are added:

pip install -r requirements.txt

⭐ 8. Troubleshooting
If Streamlit does not launch:

Try:

pip install --upgrade pip
pip install -r requirements.txt

If virtual environment is broken:

Delete the .venv folder and recreate it:

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

⭐ 9. Optional: How to Clone Using HTTPS or SSH

Both methods work.

Method 1: HTTPS (Recommended for most users)

Simple and requires no setup.

git clone https://github.com/farzadismyid/demo_wpp.git


GitHub may ask for:

Your GitHub username

A one-time Personal Access Token (instead of a password)

Method 2: SSH (For advanced users)

If you prefer passwordless Git operations.

Step 1: Generate your SSH key:
ssh-keygen -t ed25519 -C "your_email@example.com"


Press Enter for all prompts.

Step 2: Add public key to GitHub:

Copy your key:

cat ~/.ssh/id_ed25519.pub


Add it at:

https://github.com/settings/keys

Step 3: Clone using SSH:
git clone git@github.com:farzadismyid/demo_wpp.git

🎉 Everything Is Ready

You can now explore:

✔️ EDA Dashboard

streamlit run eda_app.py

✔️ Recommender Dashboard

cd app
streamlit run dashboard.py

If you need help, contact Ali.