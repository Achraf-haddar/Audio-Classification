import preprocessing
import model
import train
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    labels = ["yes", "no", "up", "down", "left", "right", 
              "on", "off", "stop", "go"]
    
    preprocessing.create_df(labels)
    df = pd.read_csv('/data/train.csv')
    # Split into train and validation set
    x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave), np.array(y),
                                            stratify=y, test_size=0.2,
                                            random_size=777, shuffle=True)
    history = train.train()