@echo off

python -m venv nlpEnv
call nlpEnv\Scripts\activate

pip install -r requirements.txt
python -m spacy download de_core_news_sm

pause