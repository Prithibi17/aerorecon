"""Download the external checkpoints expected by AeroRecon AI."""
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'Ruicheng/moge-2-vitb-normal': ROOT / 'models' / 'MoGe-2-Base',
    'nvidia/segformer-b0-finetuned-ade-512-512': ROOT / 'models' / 'SegFormer-B0',
    'depth-anything/DA3-SMALL': ROOT / 'models' / 'DA3-SMALL',
}


def main():
    for repository, destination in MODELS.items():
        print(f'Downloading {repository} -> {destination}', flush=True)
        snapshot_download(repo_id=repository, local_dir=destination)
    print('Model downloads complete.', flush=True)


if __name__ == '__main__':
    main()
