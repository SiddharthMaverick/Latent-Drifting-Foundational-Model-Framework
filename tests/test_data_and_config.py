import torch

from latent_drifting.config.loaders import load_config, save_config
from latent_drifting.config.schema import AppConfig
from latent_drifting.data.sample_queue import SampleQueue
from latent_drifting.features.pixel import PixelPatchifier


def test_sample_queue_class_isolation_and_restore():
    queue = SampleQueue(num_classes=3, class_capacity=4, unconditional_capacity=5, seed=7)
    samples = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    queue.push(samples, labels, unconditional=True)
    selected = queue.sample_class(1, 2)
    assert selected.shape == (2, 2)
    assert set(selected.flatten().tolist()).issubset(set(samples[2:4].flatten().tolist()))
    restored = SampleQueue(num_classes=3, class_capacity=4, unconditional_capacity=5, seed=99)
    restored.load_state_dict(queue.state_dict())
    assert restored.occupancy() == queue.occupancy()
    assert restored.sample_unconditional(3).shape == (3, 2)


def test_pixel_patchifier_round_trip():
    patchifier = PixelPatchifier(8)
    images = torch.randn(2, 3, 32, 48)
    reconstructed = patchifier.unpatchify(patchifier(images))
    assert torch.equal(images, reconstructed)


def test_yaml_config_round_trip(tmp_path):
    config = AppConfig()
    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.generator.num_classes == 1000
    assert loaded.drift.temperatures == (0.02, 0.05, 0.2)
