from model.gcn import GraphConvolutionalNetwork
from model.training import train
from tests.factory import generate_random_input_data, generate_random_adjacency_matrix, generate_random_data

NUM_VERTICES = 200
NUM_FEATURES = 1000
NUM_CLASSES = 4
GC_HIDDEN_SIZES = [128, 64]
FC_HIDDEN_SIZES = [32, NUM_CLASSES]
NUM_TRAINING_EXAMPLES = 100
N_EPOCHS = 3


def test_forward() -> None:
    gcn = GraphConvolutionalNetwork(
        in_features=NUM_FEATURES,
        gc_hidden_sizes=GC_HIDDEN_SIZES,
        fc_hidden_sizes=FC_HIDDEN_SIZES,
        add_residual_connection=True
    )
    input = generate_random_input_data(num_vertices=NUM_VERTICES, num_features=NUM_FEATURES)
    adjacency = generate_random_adjacency_matrix(num_vertices=NUM_VERTICES)
    output = gcn(input, adjacency)
    assert output.shape[0] == NUM_CLASSES


def test_train() -> None:
    gcn = GraphConvolutionalNetwork(
        in_features=NUM_FEATURES,
        gc_hidden_sizes=[512, 256, 128, 64],
        fc_hidden_sizes=[32, 16, 8, 4],
        softmax_outputs=True
    )
    data = generate_random_data(
        num_vertices=NUM_VERTICES,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        num_examples=NUM_TRAINING_EXAMPLES
    )
    train(
        model=gcn,
        data=data,
        num_epochs=10
    )



