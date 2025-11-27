WPP Demo Project
Analytics Dashboard and Recommendation Engine Demo

This repository contains two applications created for internal demonstration:

EDA App
A Streamlit dashboard showing exploratory data analysis.

Recommender Dashboard
An interactive dashboard that displays recommendation results, metrics, and drill-downs.

Both tools run locally and require minimal setup.

⭐ 1. Requirements

Make sure the following are installed:

Python 3.10 or newer

pip

Git

Large data files are stored externally on OneDrive.

⭐ 2. Cloning the Repository

You must be added as a collaborator.

Clone using HTTPS (recommended):
git clone https://github.com/farzadismyid/demo_wpp.git
cd demo_wpp

Clone using SSH (optional):
git clone git@github.com:farzadismyid/demo_wpp.git

⭐ 3. Download Required Data Files

Due to GitHub’s LFS limitations, large dataset files are stored on OneDrive.

Download from OneDrive:

👉 OneDrive Folder:
"Contact me for data link"

After downloading:

Place the files in these locations:

wpp_2021_2023_final.parquet → demo_wpp/
wpp_2020_2024_final.parquet → demo_wpp/
wpp_2020_2024_final.parquet → demo_wpp/app/

The applications will automatically detect these files.

⭐ 4. Create a Virtual Environment
Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

Windows:
python -m venv .venv
.venv\Scripts\activate

⭐ 5. Install Dependencies
pip install -r requirements.txt


This installs Streamlit, Pandas, DuckDB, and other required packages.

⭐ 6. Running the Applications
🔷 A) EDA App

Run from the root folder:

streamlit run eda_app.py

🔷 B) Recommender Dashboard

Run from inside the app/ folder:

cd app
streamlit run dashboard.py


🔷 C) Jupyter Notebook (Alternative to Recommendation Dashboard)

If Streamlit is slow or the dashboard freezes, you can use the Jupyter notebook version of the recommendation engine.

Notebook file:

wpp_hybrid_engine_v2.ipynb


This notebook includes:

All models used by the recommendation system

Adjustable weights and parameters

A fully repeatable workflow

To run the notebook:

Start Jupyter from the root folder:

jupyter notebook


Then open:

wpp_hybrid_engine_v2.ipynb

Required data next to the notebook:

Make sure the following file is available in the expected folders:

wpp_2020_2024_final.parquet in demo_wpp/app/

⭐ 7. Updating the Project

To pull latest changes:

git pull


If new dependencies were added:

pip install -r requirements.txt

⭐ 8. Troubleshooting
If Streamlit does not start:
pip install --upgrade pip
pip install -r requirements.txt

If the virtual environment breaks:
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

🎉 Everything is ready

You can now explore:

EDA Dashboard

Recommender Dashboard

Jupyter Notebook

Data insights and engine outputs

If anything is unclear, feel free to ask.
