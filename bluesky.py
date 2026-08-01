from graphbench.datasets import BlueSkyDataset
from pathlib import Path

root = '/gfs/shared/public/datasets/graphbench-updated/'
Path(root).mkdir(parents=True, exist_ok=True)
optional_datasets = [
    "bluesky_quotes",
    "bluesky_replies",
    "bluesky_reposts"
]
for dataset_name in optional_datasets:
    dataset = BlueSkyDataset(name=dataset_name, root=root, split='test')
    import ipdb; ipdb.set_trace()
    print(f"Dataset: {dataset}")

    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of nodes: {dataset.data.x.shape}")

    import graphbench
    Loader = graphbench.Loader(root, ['bluesky_reposts'])
    datasets = Loader.load()
