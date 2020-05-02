import argparse
import logging
from os import listdir
from os.path import isfile, join
from typing import List

import pandas as pd

from data_processing.model_data import get_model_data, partition_data
from models.gcn import GraphConvolutionalNetwork
from models.sequential_gcn import SequentialGraphConvolutionalNetwork
from models.training import train
from utils.metrics import calculate_metrics
from experiments.movie_review_dataset_constants import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    help='data directory',
    type=str,
    default=DATA_DIR
)
parser.add_argument(
    '--embeddings_path',
    help='embeddings path',
    type=str,
    default=EMBEDDINGS_PATH
)
parser.add_argument(
    '--num_trials',
    help='number of different seeds to try',
    type=str,
    default=2
)
parser.add_argument(
    '--max_epochs',
    help='max number of epochs to train for',
    type=int,
    default=NUM_EPOCHS
)
parser.add_argument(
    '--lrs',
    help='all learning rates to try',
    type=float,
    nargs="+",
    default=[LEARNING_RATE]
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
    '--is_sequential',
    help='data directory',
    type=bool,
    default=USE_SEQUENTIAL_GCN
)
parser.add_argument(
    '--gc_hidden',
    help='list of graph convolutional layer hidden sizes',
    nargs="+",
    type=int,
    action="append",
    default=[GC_HIDDEN_SIZES]
)
parser.add_argument(
    '--dropout',
    help='Dropout proportion for fully connected layers',
    nargs="+",
    type=float,
    action="append",
    default=[DROPOUT]
)
parser.add_argument(
    '--fc_hidden',
    help='list of fully connected layer hidden sizes',
    nargs="+",
    type=int,
    action="append",
    default=[FC_HIDDEN_SIZES]
)
parser.add_argument(
    '--softmax_pooling',
    help='Applies a softmax to the pooled outputs of the final GCN layer if true',
    type=bool,
    default=SOFTMAX_POOLING
)
parser.add_argument(
    '--forward_weights_size',
    help='Number of forwards weights to learn for sequential GCN model',
    type=int,
    default=FORWARD_WEIGHTS_SIZE
)
parser.add_argument(
    '--backward_weights_size',
    help='Number of backward weights to learn for sequential GCN model',
    type=int,
    default=BACKWARD_WEIGHTS_SIZE
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
    default='models'
)
args = parser.parse_args()

if __name__ == "__main__":
    metrics: List[str] = args.metrics
    valid_metrics: List[str] = ["valid_" + am for am in args.metrics]
    test_metrics: List[str] = ["test_" + am for am in args.metrics]

    class_files = [
        join(args.data_dir, f) for f in listdir(args.data_dir)
        if isfile(join(args.data_dir, f)) and (".DS_Store" not in f)
    ]

    _, data, in_features = get_model_data(
        doc_paths=class_files,
        embeddings_path=args.embeddings_path,
        max_examples_per_class=args.max_examples,
    )

    results = []
    for seed in range(args.num_trials):

        train_data, val_data, test_data = partition_data(data=data, seed=seed)

        for gc_hs in args.gc_hidden:
            for fc_hs in args.fc_hidden:
                for lr in args.lrs:
                    for dropout in args.dropout:

                        model_id = f"gc_{'-'.join(map(str, gc_hs))}_fc_{'-'.join(map(str, fc_hs))}_\
                        lr_{'{0:.0E}'.format(lr)}_dropout_{'{0:.2f}'.format(dropout).replace('.', '-')}_seed_{seed}"

                        if args.is_sequential:
                            gcn_model = SequentialGraphConvolutionalNetwork(
                                in_features=in_features,
                                gc_hidden_sizes=gc_hs,
                                fc_hidden_sizes=fc_hs,
                                forward_weights_size=args.forward_weights_size,
                                backward_weights_size=args.backward_weights_size,
                                dropout=dropout,
                                seed=seed
                            )
                            model_id += f"fws_{args.forward_weights_size}_bws_{args.backward_weights_size}"
                        else:
                            gcn_model = GraphConvolutionalNetwork(
                                in_features=in_features,
                                gc_hidden_sizes=gc_hs,
                                fc_hidden_sizes=fc_hs,
                                fc_dropout=dropout,
                                softmax_pooling=args.softmax_pooling,
                                seed=seed
                            )

                        for epoch in range(args.max_epochs):

                            train(
                                model=gcn_model,
                                train_data=train_data,
                                num_epochs=1,
                                learning_rate=lr,
                                model_path=f"{args.model_dir}{args.model_prefix}_{model_id}_epoch_{epoch}.pt"
                            )

                            train_metrics = calculate_metrics(model=gcn_model, data=val_data)
                            valid_metrics_result = calculate_metrics(model=gcn_model, data=val_data)
                            test_metrics_result = calculate_metrics(model=gcn_model, data=test_data)

                            result = {
                                "gc_hidden_layers": str(gc_hs),
                                "fc_hidden_layers": str(fc_hs),
                                "learning_rate": lr,
                                "epoch": epoch + 1,
                                "dropout": dropout,
                                "seed": seed
                            }
                            if args.is_sequential:
                                result["forward_weights_size"] = args.forward_weights_size
                                result["backward_weights_size"] = args.backward_weights_size
                            for m in metrics:
                                result[f"train_{m}"] = train_metrics[m]
                                result[f"val_{m}"] = valid_metrics_result[m]
                                result[f"test_{m}"] = test_metrics_result[m]
                            logger.info(result)
                            results.append(result)

    pd.DataFrame(results).to_csv("../../data/results/" + args.model_prefix + ".csv")
