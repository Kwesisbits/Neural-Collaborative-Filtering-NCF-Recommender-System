from tqdm import tqdm

def train_gmf(model, train_loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for batch_users, batch_items, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_users = batch_users.to(device)
            batch_items = batch_items.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            preds = model(batch_users, batch_items)

            # Loss
            loss = criterion(preds, batch_labels)
            total_loss += loss.item()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

#run training for GMF
train_gmf(model, train_loader, epochs=5)

def train_mlp(model, train_loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_users, batch_items, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_users = batch_users.to(device)
            batch_items = batch_items.to(device)
            batch_labels = batch_labels.to(device)

            preds = model(batch_users, batch_items)
            loss = criterion(preds, batch_labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
#run training for MLP
train_mlp(mlp_model, train_loader, epochs=5)

def train_neumf(model, train_loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_users, batch_items, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_users = batch_users.to(device)
            batch_items = batch_items.to(device)
            batch_labels = batch_labels.to(device)

            preds = model(batch_users, batch_items)
            loss = criterion(preds, batch_labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

  #run trainning for NeuMF 
  train_neumf(neumf_model, train_loader, epochs=5)
