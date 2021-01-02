from microservice_train_store import train
from microservice_data_processing import load_dataset

rezultate_finale = train.DT_boosted_fit(load_dataset.x_train, load_dataset.y_train, load_dataset.x_test, load_dataset.y_test, "ALL")
print(rezultate_finale)
