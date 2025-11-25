#Generalized Matrix Factorization (GMF) Model
import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim=32):
        super(GMF, self).__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)

        # Prediction layer
        self.output = nn.Linear(latent_dim, 1)

        # Sigmoid for implicit probability
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.kaiming_uniform_(self.output.weight, a=1, nonlinearity='sigmoid')
        nn.init.zeros_(self.output.bias)

    def forward(self, user_ids, item_ids):
        # Embedding lookup
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)

        # Element-wise multiplication
        x = user_vec * item_vec

        # Predict
        x = self.output(x)

        return self.sigmoid(x).squeeze()
      
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

gmf_model = GMF(num_users=num_users, num_items=num_items, latent_dim=32).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Multi-Layer Perception (MLP) Model
class MLP(nn.Module):
    def __init__(self, num_users, num_items, layers=[64,32,16,8]):
        super(MLP, self).__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, layers[0]//2)
        self.item_embedding = nn.Embedding(num_items, layers[0]//2)

        # MLP layers
        mlp_layers = []
        input_size = layers[0]
        for layer_size in layers[1:]:
            mlp_layers.append(nn.Linear(input_size, layer_size))
            mlp_layers.append(nn.ReLU())
            input_size = layer_size
        self.mlp = nn.Sequential(*mlp_layers)

        # Prediction layer
        self.output = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.kaiming_uniform_(self.output.weight, a=1, nonlinearity='sigmoid')
        nn.init.zeros_(self.output.bias)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embedding(user_ids)
        item_vec = self.item_embedding(item_ids)
        x = torch.cat([user_vec, item_vec], dim=-1)
        x = self.mlp(x)
        x = self.output(x)
        return self.sigmoid(x).squeeze()

mlp_model = MLP(num_users=num_users, num_items=num_items).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

#Neural Matrix Factorization (NeuMF) Model:
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=32, layers=[64,32,16,8]):
        super(NeuMF, self).__init__()

        # GMF embeddings
        self.gmf_user_emb = nn.Embedding(num_users, mf_dim)
        self.gmf_item_emb = nn.Embedding(num_items, mf_dim)

        # MLP embeddings
        self.mlp_user_emb = nn.Embedding(num_users, layers[0]//2)
        self.mlp_item_emb = nn.Embedding(num_items, layers[0]//2)

        # MLP layers
        mlp_layers = []
        input_size = layers[0]
        for layer_size in layers[1:]:
            mlp_layers.append(nn.Linear(input_size, layer_size))
            mlp_layers.append(nn.ReLU())
            input_size = layer_size
        self.mlp = nn.Sequential(*mlp_layers)

        # Final prediction layer
        self.output = nn.Linear(mf_dim + input_size, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for emb in [self.gmf_user_emb, self.gmf_item_emb,
                    self.mlp_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, std=0.01)
        nn.init.kaiming_uniform_(self.output.weight, a=1, nonlinearity='sigmoid')
        nn.init.zeros_(self.output.bias)

    def forward(self, user_ids, item_ids):
        # GMF branch
        gmf_u = self.gmf_user_emb(user_ids)
        gmf_i = self.gmf_item_emb(item_ids)
        gmf_out = gmf_u * gmf_i

        # MLP branch
        mlp_u = self.mlp_user_emb(user_ids)
        mlp_i = self.mlp_item_emb(item_ids)
        mlp_out = torch.cat([mlp_u, mlp_i], dim=-1)
        mlp_out = self.mlp(mlp_out)

        # Concatenate and predict
        final_input = torch.cat([gmf_out, mlp_out], dim=-1)
        x = self.output(final_input)
        return self.sigmoid(x).squeeze()
neumf_model = NeuMF(num_users=num_users, num_items=num_items).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
