def evaluate_model(model, test_user_item, user_pos_items, K=10, num_items=num_items, device=device):
    HR, NDCG = [], []
    model.eval()
    
    with torch.no_grad():
        for user, gt_item in test_user_item.items():
            # Sample 99 negatives
            neg_items = []
            while len(neg_items) < 99:
                neg = np.random.randint(0, num_items)
                if neg not in user_pos_items[user] and neg != gt_item:
                    neg_items.append(neg)
            
            item_candidates = [gt_item] + neg_items
            user_tensor = torch.tensor([user]*len(item_candidates), dtype=torch.long).to(device)
            item_tensor = torch.tensor(item_candidates, dtype=torch.long).to(device)

            # Predict
            scores = model(user_tensor, item_tensor).cpu().numpy()
            # Rank items
            ranklist = [x for _, x in sorted(zip(scores, item_candidates), reverse=True)]

            # Metrics
            HR.append(int(gt_item in ranklist[:K]))
            if gt_item in ranklist[:K]:
                index = ranklist.index(gt_item)
                NDCG.append(1 / np.log2(index + 2))
            else:
                NDCG.append(0)
                
    return np.mean(HR), np.mean(NDCG)

#Run tests for each model 
gmf_hr, gmf_ndcg = evaluate_model(gmf_model, test_user_item, user_pos_items, K=10)
print(f"GMF — HR@10: {gmf_hr:.4f}, NDCG@10: {gmf_ndcg:.4f}")

mlp_hr, mlp_ndcg = evaluate_model(mlp_model, test_user_item, user_pos_items, K=10)
print(f"MLP — HR@10: {mlp_hr:.4f}, NDCG@10: {mlp_ndcg:.4f}")

neumf_hr, neumf_ndcg = evaluate_model(neumf_model, test_user_item, user_pos_items, K=10)
print(f"NeuMF — HR@10: {neumf_hr:.4f}, NDCG@10: {neumf_ndcg:.4f}")
