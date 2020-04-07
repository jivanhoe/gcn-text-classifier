import argparse
import logging

from typing import List
from os import listdir
from os.path import isfile, join
import pandas as pd
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
    help='number of epochs',
    type=int,
    default=5
)
parser.add_argument(
    '--lrs',
    help='all learning rates to try',
    type=int,
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
    class_files = [join(args.data_dir, f) for f in listdir(args.data_dir) if isfile(join(args.data_dir, f))]

    train_data, test_data, in_features = get_model_data(
        doc_paths=class_files,
        embeddings_path=args.embeddings_path,
        max_examples_per_class=args.max_examples
    )

    results = pd.DataFrame(
        columns=['gc_hidden_layers', 'fc_hidden_layers', 'learning_rate'] + metrics
    )

    for gc_hs in args.gc_hidden:
        for fc_hs in args.fc_hidden:
            for lr in args.lrs:
                gcn_model = GraphConvolutionalNetwork(
                    in_features=in_features,
                    gc_hidden_sizes=gc_hs,
                    fc_hidden_sizes=fc_hs,
                    add_residual_connection=False
                )

                model_desc = "_gc_" + str(gc_hs) + "_fc_" + str(fc_hs) + "_lr_" + str(lr)

                train(
                    model=gcn_model,
                    train_data=train_data,
                    validation_data=test_data,
                    num_epochs=args.epochs,
                    learning_rate=lr,
                    metrics_to_log=metrics,
                    model_path=args.model_dir + args.model_prefix + ".pt"
                )

                model_metrics = calculate_metrics(model=gcn_model, data=test_data)

                result = {
                    "gc_hidden_layers": str(gc_hs),
                    "fc_hidden_layers": str(fc_hs),
                    "learning_rate": lr
                }

                for m in metrics:
                    result[m] = model_metrics[m]

                results = results.append(result, ignore_index=True)

    results.to_csv("../../data/results/" + args.model_prefix + ".csv")