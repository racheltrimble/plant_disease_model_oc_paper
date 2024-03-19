virtualenv -p `which python3.8` venv && source venv/bin/activate
pip install setuptools==65.5.0
pip install pip==21.3.1
pip install -r requirements.txt
pip install -e .
mkdir data
mkdir analysis
mkdir output
mkdir notebooks/report
mkdir notebooks/tiffs
