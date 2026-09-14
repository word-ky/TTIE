import numpy as np
import torch

from ttie.retinex_exporter import forward, parser


def test_padding_unpadding_and_clamp():
    class Probe(torch.nn.Module):
        def forward(self, x):
            self.shape = x.shape
            return x * 2 - .2
    model = Probe()
    x = torch.linspace(0, 1, 3 * 13 * 17).reshape(1, 3, 13, 17)
    got = forward(x, model)
    expected = (x * 2 - .2).clamp(0, 1)[0].permute(1, 2, 0).numpy()
    assert model.shape == (1, 3, 16, 20)
    np.testing.assert_array_equal(got, expected)


def test_native_geometry_has_no_effective_pad():
    class Probe(torch.nn.Module):
        def forward(self, x):
            assert x.shape == (1, 3, 400, 600)
            return x
    x = torch.full((1, 3, 400, 600), .25)
    np.testing.assert_array_equal(forward(x, Probe()), np.full((400, 600, 3), .25, np.float32))


def test_cli_has_only_low_checkpoint_config_output():
    assert {a.dest for a in parser()._actions} == {'help', 'low', 'checkpoint', 'config', 'out'}
