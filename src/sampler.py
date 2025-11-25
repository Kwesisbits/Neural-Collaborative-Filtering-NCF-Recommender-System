user_pos_items = (
    train_df.groupby("user_id")["item_id"]
    .apply(set)
    .to_dict()
)

#This samples items the user has never interacted with.
import random

def sample_negative_items(user, num_negatives, num_items, user_pos_dict):
  negatives = []
  positives = user_pos_dict[user]

  while len(negatives) < num_negatives:
    neg = random.randint(0, num_items-1)
    if neg not in positives:
      negatives.append(neg)

  return negatives

num_neg = 4  # standard

train_users = []
train_items = []
train_labels = []

for row in train_df.itertuples():
    u = row.user_id
    i = row.item_id

    # Add positive sample
    train_users.append(u)
    train_items.append(i)
    train_labels.append(1)

    # Add negative samples
    negs = sample_negative_items(u, num_neg, num_items, user_pos_items)
    for neg in negs:
        train_users.append(u)
        train_items.append(neg)
        train_labels.append(0)
#Convert to torch tensors
import torch

train_users = torch.tensor(train_users, dtype=torch.long)
train_items = torch.tensor(train_items, dtype=torch.long)
train_labels = torch.tensor(train_labels, dtype=torch.float32)

#Build data loader
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(train_users, train_items, train_labels)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
