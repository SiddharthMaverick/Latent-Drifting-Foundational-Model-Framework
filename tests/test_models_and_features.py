import torch

from latent_drifting.config.schema import GeneratorConfig
from latent_drifting.features.descriptors import extract_descriptors
from latent_drifting.features.hooks import FeatureEncoder
from latent_drifting.features.pixel import PixelPatchifier
from latent_drifting.mae.resnet_encoder import ResNetMAEEncoder
from latent_drifting.models.generator import DriftingGenerator


def test_generator_one_pass_output_shape_and_gradients():
    config = GeneratorConfig(
        input_channels=4,
        image_size=8,
        patch_size=2,
        hidden_dim=32,
        depth=1,
        num_heads=4,
        num_classes=3,
        register_tokens=2,
        style_tokens=2,
        style_codebook_size=4,
    )
    model = DriftingGenerator(config)
    noise = torch.randn(2, 4, 8, 8)
    labels = torch.tensor([0, 1])
    alpha = torch.tensor([1.0, 2.0])
    output = model(noise, labels, alpha, torch.zeros(2, 2, dtype=torch.long))
    assert output.shape == noise.shape
    output.square().mean().backward()
    assert model.final_layer.projection.weight.grad is not None


def test_feature_pyramid_shapes_and_frozen_encoder_input_gradient():
    encoder = ResNetMAEEncoder(input_channels=4, base_width=8, blocks_per_stage=(1, 1, 1, 1))
    extractor = FeatureEncoder(encoder)
    inputs = torch.randn(2, 4, 32, 32, requires_grad=True)
    features = extractor(inputs)
    assert "stage1_out/locations" in features
    assert "input/squared_channel_mean" in features
    sum(value.square().mean() for value in features.values()).backward()
    assert inputs.grad is not None
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_descriptor_groups_preserve_batch_and_channels():
    maps = {"stage": torch.randn(3, 8, 8, 8)}
    descriptors = extract_descriptors(maps, torch.randn(3, 4, 8, 8))
    assert descriptors["stage/locations"].shape == (3, 64, 8)
    assert descriptors["stage/patch4_std"].shape == (3, 4, 8)
    assert descriptors["input/squared_channel_mean"].shape == (3, 1, 4)


def test_pixel_patchified_encoder_backpropagates_to_pixels():
    patchifier = PixelPatchifier(8)
    encoder = ResNetMAEEncoder(input_channels=192, base_width=8, blocks_per_stage=(1, 1, 1, 1))
    extractor = FeatureEncoder(encoder, input_projection=patchifier)
    images = torch.randn(2, 3, 64, 64, requires_grad=True)
    descriptors = extractor(images)
    sum(value.square().mean() for value in descriptors.values()).backward()
    assert images.grad is not None
