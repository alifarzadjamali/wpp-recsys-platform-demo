WPP Demo Project
Analytics Dashboard and Recommendation Engine Demo

This repository contains two applications created for internal demonstration:

EDA App
A Streamlit dashboard showing exploratory data analysis of the WPP dataset.

Recommender Dashboard
An interactive dashboard that displays recommendation outputs, metrics, and drill-downs.

Both tools run locally and require minimal setup.

⭐ 1. Requirements

Please make sure the following are installed:

Python 3.10 or newer

pip

Git

Large data files are stored externally on OneDrive.

⭐ 2. Cloning the Repository

You must be added as a collaborator to access this private repo.

Clone using HTTPS (recommended):
git clone https://github.com/farzadismyid/demo_wpp.git
cd demo_wpp


SSH is also supported if you prefer it:

git clone git@github.com:farzadismyid/demo_wpp.git

⭐ 3. Download Required Data Files

Due to GitHub LFS limitations, the dataset is stored on OneDrive.

Download from OneDrive:

After downloading:

Put wpp_2021_2023_final.parquet into the root folder (demo_wpp/)

Put wpp_2020_2024_final.parquet into the app/ folder

The dashboards will automatically detect these files.

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

⭐ 7. Updating the Project

To pull latest changes:

git pull


If new dependencies were added:

pip install -r requirements.txt

⭐ 8. Troubleshooting
If Streamlit does not start:
pip install --upgrade pip
pip install -r requirements.txt

If your virtual environment breaks:
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

🎉 Everything is ready

You can now explore:

EDA Dashboard

Recommender Dashboard

Data insights and engine outputs

If anything is unclear, let me know.
