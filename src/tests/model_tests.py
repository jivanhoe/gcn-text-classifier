from models.gcn import GraphConvolutionalNetwork
from models.sequential_gcn import SequentialGraphConvolutionalNetwork
from models.training import train
from tests.factory import generate_random_input_data, generate_random_adjacency_matrix, generate_random_data

NUM_VERTICES = 20
NUM_FEATURES = 100
NUM_CLASSES = 4
GC_HIDDEN_SIZES = [128, 64]
FC_HIDDEN_SIZES = [32, NUM_CLASSES]
FORWARD_WEIGHTS_SIZE = 2
BACKWARD_WEIGHTS_SIZE = 2
NUM_TRAINING_EXAMPLES = 100
N_EPOCHS = 3


def test_gcn_forward() -> None:
    gcn_model = GraphConvolutionalNetwork(
        in_features=NUM_FEATURES,
        gc_hidden_sizes=GC_HIDDEN_SIZES,
        fc_hidden_sizes=FC_HIDDEN_SIZES,
        add_residual_connection=True,
        softmax_pooling=True
    )
    input = generate_random_input_data(num_vertices=NUM_VERTICES, num_features=NUM_FEATURES)
    adjacency = generate_random_adjacency_matrix(num_vertices=NUM_VERTICES)
    output = gcn_model(input, adjacency)
    assert output.shape[0] == NUM_CLASSES


def test_sequential_gcn_forward() -> None:
    sequential_gcn_model = SequentialGraphConvolutionalNetwork(
        in_features=NUM_FEATURES,
        gc_hidden_sizes=GC_HIDDEN_SIZES,
        fc_hidden_sizes=FC_HIDDEN_SIZES,
        forward_weights_size=FORWARD_WEIGHTS_SIZE,
        backward_weights_size=BACKWARD_WEIGHTS_SIZE,
        add_residual_connection=True,
        softmax_pooling=True
    )
    input = generate_random_input_data(num_vertices=NUM_VERTICES, num_features=NUM_FEATURES)
    output = sequential_gcn_model(input)
    assert output.shape[0] == NUM_CLASSES


def test_train() -> None:
    gcn_model = GraphConvolutionalNetwork(
        in_features=NUM_FEATURES,
        gc_hidden_sizes=GC_HIDDEN_SIZES,
        fc_hidden_sizes=FC_HIDDEN_SIZES
    )
    sequential_gcn_model = SequentialGraphConvolutionalNetwork(
        in_features=NUM_FEATURES,
        gc_hidden_sizes=GC_HIDDEN_SIZES,
        fc_hidden_sizes=FC_HIDDEN_SIZES,
        forward_weights_size=FORWARD_WEIGHTS_SIZE,
        backward_weights_size=BACKWARD_WEIGHTS_SIZE,
        add_residual_connection=True,
        softmax_pooling=True
    )
    train_data = generate_random_data(
        num_vertices=NUM_VERTICES,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        num_examples=NUM_TRAINING_EXAMPLES
    )
    validation_data = generate_random_data(
        num_vertices=NUM_VERTICES,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        num_examples=NUM_TRAINING_EXAMPLES
    )
    train(
        model=gcn_model,
        train_data=train_data,
        validation_data=validation_data,
        num_epochs=N_EPOCHS
    )
    train(
        model=sequential_gcn_model,
        train_data=train_data,
        validation_data=validation_data,
        num_epochs=N_EPOCHS
    )



