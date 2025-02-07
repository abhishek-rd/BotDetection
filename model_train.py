import torch
import networkx as nx
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load data
val_idx = torch.load('/content/val_idx.pt')
tweets_tensor = torch.load('/content/tweets_tensor.pt')
train_idx = torch.load('/content/train_idx.pt')
test_idx = torch.load('/content/test_idx.pt')
num_properties_tensor = torch.load('/content/num_properties_tensor.pt')
label = torch.load('/content/label.pt')
edge_index = torch.load('/content/edge_index.pt')


# Define the GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x



in_channels = num_properties_tensor.shape[1]
out_channels = 128
model = GraphSAGE(in_channels, out_channels)
data = Data(x=num_properties_tensor, edge_index=edge_index)

# Generate GraphSAGE embeddings
model.eval()
with torch.no_grad():
    graphSAGE_embeddings = model(data.x, data.edge_index)

# Concatenate embeddings
combined_embeddings = torch.cat((graphSAGE_embeddings, tweets_tensor), dim=1)


labels = label.numpy()
combined_embeddings = combined_embeddings.numpy()

# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []
fold = 1

for train_index, test_index in kf.split(combined_embeddings):
    print(f"Fold {fold}:")
    fold += 1


    X_train, X_test = combined_embeddings[train_index], combined_embeddings[test_index]
    y_train, y_test = labels[train_index], labels[test_index]


    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)


    y_pred = svm_classifier.predict(X_test)


    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print("Accuracy for this fold:", accuracy)


print("\nSummary:")
print("Cross-Validation Accuracy Scores:", accuracy_scores)
print("Average Accuracy:", sum(accuracy_scores) / len(accuracy_scores))
