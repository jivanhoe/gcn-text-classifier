# Paths
DATA_DIR = "../../data/movie_reviews/"
POSITIVE_REVIEWS_PATH = f"{DATA_DIR}positive_reviews.txt"
NEGATIVE_REVIEWS_PATH = f"{DATA_DIR}negative_reviews.txt"
EMBEDDINGS_PATH = "../../data/glove/glove_6b_300d.txt"
MODEL_PATH = "../../data/models/movie_reviews_model.pt"
CLUSTERING_DATA_DIR = "../../data/clustering/"

# Model parameters
GC_HIDDEN_SIZES = [128, 128]
FC_HIDDEN_SIZES = [64, 2]  # Final fully-connected layer size must equal number of classes
FORWARD_WEIGHTS = [0.5, 0.25, 0.125]
BACKWARD_WEIGHTS = [0.5, 0.25, 0.125]
FORWARD_WEIGHTS_SIZE = len(FORWARD_WEIGHTS)
BACKWARD_WEIGHTS_SIZE = len(BACKWARD_WEIGHTS)
DROPOUT = 0.4
IN_FEATURES = 300
ADD_RESIDUAL_CONNECTION = False
SOFTMAX_POOLING = True
USE_CUSTOM_ADJACENCY_MATRIX = True
USE_SEQUENTIAL_GCN = True

# Training parameters
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_EXAMPLES_PER_CLASS = None
METRICS_TO_LOG = ["accuracy", "auc", "f1"]

# Misc
SEED = 0
