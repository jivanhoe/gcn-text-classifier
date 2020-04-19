# Paths
DATA_DIR = "../../data/movie_reviews/"
POSITIVE_REVIEWS_PATH = f"{DATA_DIR}positive_reviews.txt"
NEGATIVE_REVIEWS_PATH = f"{DATA_DIR}negative_reviews.txt"
EMBEDDINGS_PATH = "../../data/glove/glove_6B_300d.txt"
MODEL_PATH = "../../data/models/movie_reviews_model.pt"
CLUSTERING_DATA_DIR = "../../data/clustering/"

# Model parameters
GC_HIDDEN_SIZES = [128, 128]
FC_HIDDEN_SIZES = [64, 2]  # Final fully-connected layer size must equal number of classes
FORWARD_WEIGHTS_SIZE = 1
BACKWARD_WEIGHTS_SIZE = 1
IN_FEATURES = 300
ADD_RESIDUAL_CONNECTION = False
SOFTMAX_POOLING = False
USE_SEQUENTIAL_GCN = False

# Training parameters
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
MAX_EXAMPLES_PER_CLASS = None
METRICS_TO_LOG = ["accuracy", "auc"]

# Misc
SEED = 0
