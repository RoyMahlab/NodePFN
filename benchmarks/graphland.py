import sys
from pathlib import Path

Datasets = ['hm-categories', 'pokec-regions', 
            'tolokers-2', 'city-reviews', 'artnet-exp']

# The graphland dataset repo isn't a package (no __init__.py anywhere up the
# tree), so we put its root on sys.path and import its modules by name.
# dataset.py locates its `data/` dir relative to its own __file__, so it works
# regardless of where we import it from.
GRAPHLAND_ROOT = Path("/gfs/shared/public/datasets/graphland")
if str(GRAPHLAND_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHLAND_ROOT))

from dataset import PyGDataset

dataset = PyGDataset(name='artnet-views', split='RL')
print(f"Loaded dataset with {len(dataset)} graphs, each with {dataset.num_features} features and {dataset.num_classes} classes.")