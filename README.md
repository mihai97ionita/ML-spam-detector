#DM

###Requirements:
- Anaconda
- Pip 

###Set up:
- Install requirements from file `pip install -r requirements.txt`

###Run scripts:
- Run web page service with `python web_page.py`
- If you want to run a training session `python main_train.py`, logged results will be found in `/train_result` and the new `/models` in models folder
- If you want to change the model used for prediction, change first line of  `/models/ACTIVE` with another model's file name found in folder `/models`