import argparse
import logging

from typing import List
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split

from data_processing.model_data import get_model_data
from model.gcn import GraphConvolutionalNetwork
from model.training import train
from utils.metrics import calculate_metrics

logger = logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    help='data directory',
    type=str,
    default="../../data/movie_reviews/"
)
parser.add_argument(
    '--embeddings_path',
    help='embeddings path',
    type=str,
    default="../../data/glove/glove_6b_300d.txt"
)
parser.add_argument(
    '--epochs',
    help='numbers of epochs',
    type=int,
    nargs="+",
    default=[5]
)
parser.add_argument(
    '--lrs',
    help='all learning rates to try',
    type=float,
    nargs="+",
    default=[2e-4]
)
parser.add_argument(
    '--max_examples',
    help='max examples per class',
    default=None
)
parser.add_argument(
    '--metrics',
    help='metrics to record',
    type=str,
    nargs="+",
    default=["accuracy", "auc"]
)
parser.add_argument(
    '--gc_hidden',
    help='list of graph convolutional layer hidden sizes',
    nargs="+",
    type=int,
    action="append",
    default=[[256, 128]]
)
parser.add_argument(
    '--fc_hidden',
    help='list of fully connected layer hidden sizes',
    nargs="+",
    type=int,
    action="append",
    default=[[64, 2]]
)
parser.add_argument(
    '--model_dir',
    help='directory to save models in',
    type=str,
    default="../../data/models/"
)
parser.add_argument(
    '--model_prefix',
    help='prefix with which to save all models',
    type=str,
    default='model'
)
args = parser.parse_args()


if __name__ == "__main__":
    metrics: List[str] = args.metrics
    valid_metrics: List[str] = ["valid_" + am for am in args.metrics]
    test_metrics: List[str] = ["test_" + am for am in args.metrics]

    class_files = [join(args.data_dir, f) for f in listdir(args.data_dir) if isfile(join(args.data_dir, f))]

    train_data, test_data, in_features = get_model_data(
        doc_paths=class_files,
        embeddings_path=args.embeddings_path,
        max_examples_per_class=args.max_examples
    )

    train_data, valid_data = train_test_split(train_data, test_size=0.3)

    results = pd.DataFrame(
        columns=['gc_hidden_layers', 'fc_hidden_layers', 'learning_rate'] + valid_metrics + test_metrics
    )

    for gc_hs in args.gc_hidden:
        for fc_hs in args.fc_hidden:
            for lr in args.lrs:
                for epochs in args.epochs:
                    gcn_model = GraphConvolutionalNetwork(
                        in_features=in_features,
                        gc_hidden_sizes=gc_hs,
                        fc_hidden_sizes=fc_hs,
                        add_residual_connection=False
                    )

                    model_desc = "_gc_" + str(gc_hs) + "_fc_" + str(fc_hs) + "_lr_" + str(lr) + "_epochs_" + str(epochs)

                    train(
                        model=gcn_model,
                        train_data=train_data,
                        validation_data=valid_data,
                        num_epochs=epochs,
                        learning_rate=lr,
                        metrics_to_log=metrics,
                        model_path=args.model_dir + args.model_prefix + ".pt"
                    )

                    valid_metrics_result = calculate_metrics(model=gcn_model, data=valid_data)
                    test_metrics_result = calculate_metrics(model=gcn_model, data=test_data)

                    result = {
                        "gc_hidden_layers": str(gc_hs),
                        "fc_hidden_layers": str(fc_hs),
                        "learning_rate": lr
                    }

                    for m in metrics:
                        result["test_" + m] = test_metrics_result[m]
                        result["valid_" + m] = valid_metrics_result[m]

                    results = results.append(result, ignore_index=True)

    results.to_csv("../../data/results/" + args.model_prefix + ".csv")
