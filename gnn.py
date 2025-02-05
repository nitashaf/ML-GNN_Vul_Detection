#imports
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_mean_pool, SAGEConv, BatchNorm, ClusterGCNConv
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
import pickle
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import numpy as np
import graph_extraction

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import KFold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def get_graph(filename):
    with open(filename, 'rb') as f:
        graph = pickle.load(f)
        return graph


from networkx.algorithms.clique import node_clique_number


# input is 7 dimensions
def nx_to_pyg(graph, label, node_label_encoder):
    # print('graph generated')
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(graph.nodes())}

    node_labels = [graph.nodes[node]['label'] for node in graph.nodes]
    node_label_features = node_label_encoder.transform(node_labels)

    edge_index = torch.tensor([(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in graph.edges],
                              dtype=torch.long).t().contiguous()

    in_degrees = [graph.in_degree(node) for node in graph.nodes]
    out_degrees = [graph.out_degree(node) for node in graph.nodes]
    degrees = [graph.degree(node) for node in graph.nodes]

    components = list(nx.strongly_connected_components(graph))
    component_sizes = {node: len(comp) for comp in components for node in comp}

    pagerank = nx.pagerank(graph, alpha=0.85)
    pagerank_values = [pagerank[node] for node in graph.nodes]

    undirected_graph = graph.to_undirected()

    clique_numbers = node_clique_number(undirected_graph)

    node_continuous_features = np.vstack([in_degrees, out_degrees, degrees,
                                          [component_sizes[node] for node in graph.nodes],
                                          list(clique_numbers.values()), pagerank_values]).T

    x = torch.cat([torch.tensor(node_label_features, dtype=torch.long).unsqueeze(1),
                   torch.tensor(node_continuous_features, dtype=torch.float)], dim=1)

    data = Data(
        x=x.to(device),
        edge_index=edge_index.to(device),
        y=torch.tensor(label, dtype=torch.long, device=device)
    )

    return data


def encode_node_labels(graphs):
    all_node_labels = []
    for graph in graphs:
        all_node_labels.extend([graph.nodes[node]['label'] for node in graph.nodes])
    label_encoder = LabelEncoder()
    label_encoder.fit(all_node_labels)
    return label_encoder


def load_dataset(root_dir):
    graphs = []
    labels = []

    cwe_to_parent = {
        "CWE122": "CWE122",
        "CWE121": "CWE121",
        "CWE124": "CWE124",
        "CWE126": "CWE126",
        "CWE127": "CWE127",
        "CWE590": "CWE590",
        "CWE415": "CWE415",
        "CWE762": "CWE762",
        "CWE789": "CWE789",
        "CWE190": "CWE190",
        "CWE191": "CWE191",
        "CWE195": "CWE195",
        "CWE194": "CWE194",
        "CWE197": "CWE197",
        "CWE680": "CWE680",
        "CWE369": "CWE369",
        "CWE134": "CWE134",
        "CWE78": "CWE78",
        "CWE36": "CWE36",
        "CWE23": "CWE23",
        "CWE401": "CWE401",
        "CWE690": "CWE690",
        "CWE400": "CWE400",
        "CWE457": "CWE457",
        "CWE758": "CWE758"
    }

    class_map = {}
    class_idx = 0

    for class_folder in os.listdir(root_dir):
        if class_folder == 'not_vulnerable':
            continue

        class_path = os.path.join(root_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        if cwe_to_parent[class_folder] not in class_map:
            class_map[cwe_to_parent[class_folder]] = class_idx
            class_idx += 1

        for graph_file in os.listdir(class_path):
            graph_path = os.path.join(class_path, graph_file)
            graph = get_graph(graph_path)
            graphs.append(graph)
            labels.append(class_map[cwe_to_parent[class_folder]])
        print(f"{class_folder} nx graphs loaded into memory, labeled as {cwe_to_parent[class_folder]}")

    return graphs, labels, class_map


def load_binary_dataset(root_dir):
    graphs = []
    labels = []
    class_map = {'not_vulnerable': 0, 'vulnerable': 1}

    for class_folder in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        label = 1 if class_folder != 'not_vulnerable' else 0
        for graph_file in os.listdir(class_path):
            graph_path = os.path.join(class_path, graph_file)
            graph = get_graph(graph_path)
            graphs.append(graph)
            labels.append(label)
        print(f"{class_folder} nx graphs loaded into memory")
    return graphs, labels, class_map


def load_data(graphs, labels, node_label_encoder):
    pyg_graphs = [nx_to_pyg(graph, label, node_label_encoder) for graph, label in zip(graphs, labels)]
    return pyg_graphs


def compute_class_weights(labels):
    labels = np.array(labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)


class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.global_pool = global_mean_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        x = self.global_pool(x, batch)
        return x


def augment_graph(graph):
    noise = np.random.normal(0, 0.01, size=graph.x.shape)
    graph.x += torch.tensor(noise, dtype=torch.float).to(device)

    mask = np.random.binomial(1, 0.1, size=graph.x.shape).astype(bool)
    graph.x[mask] = 0

    # # Randomly add/remove edges
    # num_edges = graph.edge_index.shape[1]
    # num_additional_edges = int(0.1 * num_edges)
    # additional_edges = torch.randint(0, graph.x.shape[0], (2, num_additional_edges)).to(device)
    # graph.edge_index = torch.cat([graph.edge_index, additional_edges], dim=1)

    # # Randomly drop edges
    # drop_mask = torch.rand(graph.edge_index.shape[1]) > 0.1
    # graph.edge_index = graph.edge_index[:, drop_mask]

    return graph


def augment_dataset(graphs):
    augmented_graphs = []
    for graph in graphs:
        augmented_graphs.append(augment_graph(graph))
    return augmented_graphs


def cross_validate(model_class, pyg_graphs, labels, num_classes, class_weights, k=5, epochs=100, batch_size=16):
    kf = KFold(n_splits=k, shuffle=True)
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    best_f1 = 0
    best_model_state = None

    for train_index, val_index in kf.split(pyg_graphs):
        train_graphs = [pyg_graphs[i] for i in train_index]
        val_graphs = [pyg_graphs[i] for i in val_index]
        train_labels = [labels[i] for i in train_index]
        val_labels = [labels[i] for i in val_index]

        model = model_class(in_channels=7, hidden_channels=128, out_channels=num_classes, dropout_rate=0.5).to(device)
        accuracy, precision, recall, f1, model_state = train_and_validate(model, train_graphs, val_graphs, device,
                                                                          num_classes, class_weights, epochs,
                                                                          batch_size)


        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model_state

        # Collect metrics
        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

        # Save the best model to the specified directory
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(best_model_state,
                   f"models/r_returned_best_model_{accuracy}_{precision}_{recall}_{best_f1}.pth")

    print(
        f"Cross-Validation Results: Accuracy: {np.mean(all_accuracies):.2f}%, Precision: {np.mean(all_precisions):.4f}, Recall: {np.mean(all_recalls):.4f}, F1 Score: {np.mean(all_f1s):.4f}")


def train_and_validate(model, train_graphs, valid_graphs, device, num_classes, class_weights, epochs=100,
                       batch_size=16):
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = None
    if class_weights is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_f1 = 0
    best_model_state = None

    for epoch in range(epochs):
        # Train the model
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Adjust learning rate based on schedule
        scheduler.step()

        # Validate the model after each epoch
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                out = model(batch)
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        # Calculate precision, recall, F1 score, and accuracy
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, "
              f"Validation Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, "
              f"Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        # Save the best model based on F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = model.state_dict()
            torch.save(best_model_state,
                       f"models/cwe_best_model_epoch_{epoch + 1}_{accuracy:.2f}_{precision:.4f}_{recall:.4f}_{f1:.4f}.pth")

    return accuracy, precision, recall, best_f1, best_model_state


def cwe_classification():
    train_graphs, train_labels, class_map = load_dataset("GraphData/for_transfer/ASTCFGDDG/training")
    valid_graphs, valid_labels, _ = load_dataset("GraphData/for_transfer/ASTCFGDDG/validating")
    node_label_encoder = encode_node_labels(train_graphs)
    pyg_train_graphs = load_data(train_graphs, train_labels, node_label_encoder)
    pyg_valid_graphs = load_data(valid_graphs, valid_labels, node_label_encoder)
    augmented_train_graphs = augment_dataset(pyg_train_graphs)
    augmented_valid_graphs = augment_dataset(pyg_valid_graphs)
    input_graphs = augmented_train_graphs + augmented_valid_graphs
    input_labels = train_labels + valid_labels
    class_weights = compute_class_weights(train_labels)
    cross_validate(GraphSAGEModel, input_graphs, input_labels, len(class_map), class_weights, k=5, epochs=50,
                   batch_size=32)


# cwe_classification()


def run_binary_classification():
    train_graphs, train_labels, _ = load_binary_dataset("GraphData/for_transfer/ASTCFGDDG/training")
    valid_graphs, valid_labels, _ = load_binary_dataset("GraphData/for_transfer/ASTCFGDDG/validating")
    node_label_encoder = encode_node_labels(train_graphs)
    pyg_train_graphs = load_data(train_graphs, train_labels, node_label_encoder)
    pyg_valid_graphs = load_data(valid_graphs, valid_labels, node_label_encoder)
    augmented_train_graphs = augment_dataset(pyg_train_graphs)
    augmented_valid_graphs = augment_dataset(pyg_valid_graphs)
    input_graphs = augmented_train_graphs + augmented_valid_graphs
    input_labels = train_labels + valid_labels
    class_weights = compute_class_weights(train_labels)
    cross_validate(GraphSAGEModel, input_graphs, input_labels, 2, None, k=5, epochs=50,
                   batch_size=32)

# run_binary_classification()


def load_model(model_path, model_class, in_channels, hidden_channels, out_channels, dropout_rate):
    model = model_class(in_channels, hidden_channels, out_channels, dropout_rate).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate_model(model, test_graphs):
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    return all_labels, all_preds


def print_classification_report(labels, preds, class_map):
    report = classification_report(labels, preds, target_names=list(class_map.keys()), zero_division=1)
    print(report)


def run_test_evaluation():
    # Load the test dataset
    test_graphs, test_labels, class_map = load_binary_dataset("GraphData/CFGDDGAST/testing")
    node_label_encoder = encode_node_labels(test_graphs)
    pyg_test_graphs = load_data(test_graphs, test_labels, node_label_encoder)
    # augmented_test_graphs = augment_dataset(pyg_test_graphs)
    augmented_test_graphs = pyg_test_graphs
    print(class_map)

    # Load the saved model
    model_path = 'models/r_returned_best_model_98.20304799454091_0.9820608385479849_0.982030479945409_0.9821823556230147.pth'
    model = load_model(model_path, GraphSAGEModel, in_channels=7, hidden_channels=128, out_channels=2, dropout_rate=0.5)

    # Evaluate the model on the test data
    labels, preds = evaluate_model(model, augmented_test_graphs)

    # Print the classification report
    unique_labels = set(labels)
    if len(unique_labels) == 1:
        print("Only one class present in the labels. Adjusting the classification report.")
        report = classification_report(labels, preds, labels=[0, 1], target_names=["non_vulnerable", "vulnerable"],
                                       zero_division=1)
    else:
        report = classification_report(labels, preds, target_names=list(class_map.keys()), zero_division=1)
    print(report)

# run_test_evaluation()

def run_test_cwe():
    test_graphs, test_labels, class_map = load_dataset("GraphData/for_transfer/ASTCFGDDG/testing")
    node_label_encoder = encode_node_labels(test_graphs)
    pyg_test_graphs = load_data(test_graphs, test_labels, node_label_encoder)
    model_path = 'models/r_returned_best_model_39.42657766990291_0.5876416639510493_0.3942657766990291_0.5308643673900084.pth'
    model = load_model(model_path, GraphSAGEModel, in_channels=7, hidden_channels=128, out_channels=len(class_map),
                       dropout_rate=0.5)
    labels, preds = evaluate_model(model, pyg_test_graphs)
    print_classification_report(labels, preds, class_map)


# run_test_cwe()


def run_single_binary(file_path, vulnerable_file_path, cwe_model_path):
    cwe_to_parent = {
        "CWE122": "CWE122",
        "CWE121": "CWE121",
        "CWE124": "CWE124",
        "CWE126": "CWE126",
        "CWE127": "CWE127",
        "CWE590": "CWE590",
        "CWE415": "CWE415",
        "CWE762": "CWE762",
        "CWE789": "CWE789",
        "CWE190": "CWE190",
        "CWE191": "CWE191",
        "CWE195": "CWE195",
        "CWE194": "CWE194",
        "CWE197": "CWE197",
        "CWE680": "CWE680",
        "CWE369": "CWE369",
        "CWE134": "CWE134",
        "CWE78": "CWE78",
        "CWE36": "CWE36",
        "CWE23": "CWE23",
        "CWE401": "CWE401",
        "CWE690": "CWE690",
        "CWE400": "CWE400",
        "CWE457": "CWE457",
        "CWE758": "CWE758"
    }
    try:
        graph = graph_extraction.extract_cfg_ddg_ast(file_path)
    except Exception as e:
        print(f'Error Loading File {file_path}: {e}')
        return f'Error Loading File {file_path}: {e} \n'
    # label as 0 is a dummy label
    if len(graph.edges()) == 0:
        print(f"Skipping {file_path}: Graph has no edges")
        return f"Skipping {file_path}: Graph has no edges \n"
    node_label_encoder = encode_node_labels([graph])
    pyg_graph = nx_to_pyg(graph, label=0, node_label_encoder=node_label_encoder)
    if pyg_graph.edge_index.size(1) == 0:
        print(f"Skipping {file_path}: No edges in the graph (empty edge_index)")
        return f"Skipping {file_path}: No edges in the graph (empty edge_index) \n"
    pyg_graph = pyg_graph.to(device)
    cwe_model = load_model(cwe_model_path, GraphSAGEModel, in_channels=7, hidden_channels=128, out_channels=25,
                       dropout_rate=0.5)
    vulnerable_model = load_model(vulnerable_file_path, GraphSAGEModel, in_channels=7, hidden_channels=128, out_channels=2, dropout_rate=0.5)
    try:
        vulnerable_model.eval()
        with torch.no_grad():
            vulnerable_output = vulnerable_model(pyg_graph)
            vulnerable_probs = F.softmax(vulnerable_output, dim=1)
            vulnerable_pred = vulnerable_probs.argmax(dim=1).item()
            vulnerable_label = "vulnerable" if vulnerable_pred == 1 else "not vulnerable"
            vulnerable_confidence = vulnerable_probs[0, vulnerable_pred].item() * 100

        cwe_model.eval()
        with torch.no_grad():
            cwe_output = cwe_model(pyg_graph)
            cwe_probs = F.softmax(cwe_output, dim=1)
            cwe_pred = cwe_probs.argmax(dim=1).item()
            cwe_class_label = list(cwe_to_parent.keys())[cwe_pred]
            cwe_confidence = cwe_probs[0, cwe_pred].item() * 100
        print(f"Processing {file_path} on both models")
        print(f'Vulnerability Status: {vulnerable_label} ({vulnerable_confidence:.2f}% confidence)')
        print(f'predicted_CWE: {cwe_class_label} ({cwe_confidence:.2f}% confidence)')

        return f'Processed {file_path} on both models \nVulnerability Status: {vulnerable_label} ({vulnerable_confidence:.2f}% confidence) \npredicted_CWE: {cwe_class_label} ({cwe_confidence:.2f}% confidence)\n\n'
    finally:
        del cwe_model, vulnerable_model, pyg_graph
        torch.cuda.empty_cache() 
        torch.cuda.synchronize()


if __name__ == '__main__':
    ''' !!!! IMPORTANT !!!!
        run_test_cwe() and run_test_evaluation() will only work if you have the juliet sard dataset downloaded and loaded properly!
        Due to the size of the juliet sard dataset it will not be included in the source but run_single_binary function should be useable
    '''
    ## This will run cwe classification based on the models in package
    # run_test_cwe()

    ## This will run binary classification based on the models in the package
    # run_test_evaluation()

    ## To run a single file use
    vuln_model = 'models/ASTCFGDDG/r_returned_best_model_98.20304799454091_0.9820608385479849_0.982030479945409_0.9821823556230147.pth'
    cwe_model = 'models/ASTCFGDDG/r_returned_best_model_49.78006977096921_0.6642518318883048_0.4978006977096921_0.5308643673900084.pth'
    binary_path = 'Insert path here'
    strings = run_single_binary(binary_path, vuln_model, cwe_model)