import torch
import networkx as nx
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

val_idx = torch.load('/content/val_idx.pt')
tweets_tensor = torch.load('/content/tweets_tensor.pt')
train_idx = torch.load('/content/train_idx.pt')
test_idx = torch.load('/content/test_idx.pt')
num_properties_tensor = torch.load('/content/num_properties_tensor.pt')
label = torch.load('/content/label.pt')
edge_type = torch.load('/content/edge_type.pt')
edge_index = torch.load('/content/edge_index.pt')
des_tensor = torch.load('/content/des_tensor.pt')
cat_properties_tensor = torch.load('/content/cat_properties_tensor.pt')

file_info = {
    "val_idx": val_idx.shape,
    "tweets_tensor": tweets_tensor.shape,
    "train_idx": train_idx.shape,
    "test_idx": test_idx.shape,
    "num_properties_tensor": num_properties_tensor.shape,
    "label": label.shape,
    "edge_type": edge_type.shape,
    "edge_index": edge_index.shape,
    "des_tensor": des_tensor.shape,
    "cat_properties_tensor": cat_properties_tensor.shape
}


edge_index = torch.load('/content/edge_index.pt')

G = nx.DiGraph()

for i in range(edge_index.shape[1]):
    source_node = edge_index[0, i].item()  # User
    target_node = edge_index[1, i].item()  # Follower
    G.add_edge(source_node, target_node)

# Define the GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

# Initialize the model with num_properties_tensor as input features
in_channels = num_properties_tensor.shape[1]
out_channels = 128
model = GraphSAGE(in_channels, out_channels)
data = Data(x=num_properties_tensor, edge_index=edge_index)

# Generate embeddings
model.eval()
with torch.no_grad():
    embeddings = model(data.x, data.edge_index)

# Concatenate GraphSAGE embeddings with BERT embeddings
combined_embeddings = torch.cat((graphSAGE_embeddings, tweets_tensor), dim=1)


X_train, X_test, y_train, y_test = train_test_split(combined_embeddings.numpy(), labels.numpy(), test_size=0.2, random_state=42)


svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)


y_pred = svm_classifier.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred,digits=4))
print("Accuracy:", accuracy_score(y_test, y_pred))


X_train, X_test, y_train, y_test = train_test_split(combined_embeddings.numpy(), labels.numpy(), test_size=0.2, random_state=42)


svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)


y_pred = svm_classifier.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))