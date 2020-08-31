from preprocessing import create_df
import model
from train import train
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot 

if __name__ == "__main__":
    labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    x_tr, x_val, y_tr, y_val = create_df(labels)
    #print(x_tr.shape)
    history = train(x_tr, x_val, y_tr, y_val, labels)

    pyplot.plot(history.history['loss'], label='train') 
    pyplot.plot(history.history['val_loss'], label='test') 
    pyplot.legend() 
    pyplot.show()
