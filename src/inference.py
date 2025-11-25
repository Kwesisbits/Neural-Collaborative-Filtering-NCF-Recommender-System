def recommend_for_user(neumf_model, user_id, top_k=10, num_items=num_items):
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_id]*num_items)
        item_tensor = torch.arange(num_items)
        scores = model(user_tensor, item_tensor).squeeze()
        top_items = torch.topk(scores, top_k).indices.tolist()
    return top_items

#Test Recommender System
top_items = recommend_for_user(neumf_model, user_id=5, top_k=10)
print(top_items)
