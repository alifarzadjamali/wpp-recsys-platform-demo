WPP Demo Project
Analytics Dashboard and Recommendation System Demo

This repository contains two small applications created for internal demonstration:

EDA App
A Streamlit dashboard for exploratory data analysis of the WPP dataset.

Recommender Dashboard
An interactive dashboard that showcases the behavior and outputs of the recommendation engine.

Both tools are self-contained and can be run locally with minimal setup.

⭐ 1. Requirements

Please make sure the following are installed:

✔ Python 3.10 or newer
python3 --version

✔ pip
pip --version

✔ Git and Git LFS

Large .parquet files are managed using Git LFS.
Cloning this repository will automatically fetch them.

⭐ 2. Cloning the Repository

If you have been added as a collaborator, you can download the project using:

HTTPS (recommended for most users):
git clone https://github.com/farzadismyid/demo_wpp.git


or SSH (optional, for advanced users):

git clone git@github.com:farzadismyid/demo_wpp.git


Then move into the folder:

cd demo_wpp

⭐ 3. Creating a Virtual Environment

Creating a virtual environment ensures that dependencies remain isolated.

Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

Windows (PowerShell):
python -m venv .venv
.venv\Scripts\activate

⭐ 4. Installing Dependencies

Install all required packages:

pip install -r requirements.txt


This will install Streamlit, Pandas, DuckDB and other dependencies.
Git LFS will automatically pull the large data files.

⭐ 5. Running the Applications

There are two independent dashboards.

🔷 A) EDA App

From the root folder:

streamlit run eda_app.py


This launches an interactive browser window with:

Usage and behavioral statistics

Category summaries

Merchant-level insights

Time-series trends

Interactive filters

🔷 B) Recommender Dashboard

This one must be run from inside the app/ directory because data files are located there.

cd app
streamlit run dashboard.py


This dashboard includes:

Global recommendation metrics

Precision@K, MAP@K, NDCG@K

User-level recommendations

Merchant information

Category drill-downs

Session-based analysis

Filtering by merchant, user, category or period

⭐ 6. Project Structure
demo_wpp/
│
├── eda_app.py                       # EDA dashboard
├── requirements.txt                 # Python dependencies
├── wpp_2021_2023_final.parquet      # Data (Git LFS)
│
├── app/
│   ├── dashboard.py                 # Recommender dashboard
│   ├── wpp_2020_2024_final.parquet  # Data (Git LFS)
│   ├── *.csv                        # Additional data files
│   └── ...
│
└── engine/
    └── Backend engine modules

⭐ 7. Updating the Project

To pull new updates:

git pull


If new packages were added:

pip install -r requirements.txt

⭐ 8. Troubleshooting
If Streamlit doesn’t open
pip install --upgrade pip
pip install -r requirements.txt

If the virtual environment breaks
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

⭐ 9. Optional: Cloning Using HTTPS or SSH
Method 1: HTTPS (easiest)
git clone https://github.com/farzadismyid/demo_wpp.git


GitHub may ask for your username and a Personal Access Token (PAT).

Method 2: SSH (optional)

Generate a key:

ssh-keygen -t ed25519 -C "your_email@example.com"


Copy your public key:

cat ~/.ssh/id_ed25519.pub


Add it to GitHub under:
https://github.com/settings/keys

Clone using SSH:

git clone git@github.com:farzadismyid/demo_wpp.git

🎉 Everything is ready

You can now explore:

EDA Dashboard
streamlit run eda_app.py

Recommender Dashboard
cd app
streamlit run dashboard.py


If anything is unclear or you want additional features, feel free to ask.