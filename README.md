# gcn-text-classifier
Graph convolutional network for text  classification

# script for cross_validation

Here is an example of how to run the cross_validation script with different learning rates, layers, etc.
`python3 cross_validation.py --epochs 8 --lrs 0.0001 0.0002 0.001 --gc_hidden 256 128 --gc_hidden 128 128 --gc_hidden 64 128 --gc_hidden 256 256 --fc_hidden 64 2 --fc_hidden 128 2 --fc_hidden 256 2`
