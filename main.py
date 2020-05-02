from functions import train
from functions import load_dataset

print("FOR ALL \n")
rezultate_finale=train.DT_boosted_fit(load_dataset.x_train, load_dataset.y_train, load_dataset.x_test, load_dataset.y_test, "ALL")
print(rezultate_finale)