
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_movielens(path):
    df = pd.read_csv(path, sep="::", engine="python", names=["user_id","item_id","rating","timestamp"])
    df["label"] = (df["rating"] >= 4).astype(int)
    
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    df["user_id"] = user_encoder.fit_transform(df["user_id"])
    df["item_id"] = item_encoder.fit_transform(df["item_id"])
    
    num_users = df["user_id"].nunique()
    num_items = df["item_id"].nunique()
    
    # Leave-one-out split
    train_data = []
    test_data = []
    for user, group in df.groupby("user_id"):
        group = group.sort_values("item_id")
        test_data.append(group.iloc[-1])
        train_data.append(group.iloc[:-1])
    train_df = pd.concat(train_data)
    test_df = pd.DataFrame(test_data)
    
    return train_df, test_df, num_users, num_items

#Save the cleaned data
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
