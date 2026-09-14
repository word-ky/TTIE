import torch
from ttie.snr_exporter import pad16, parser, snr_mask


def test_native_pad16_is_right_eight_columns_and_unpad_exact():
    x = torch.arange(400 * 600, dtype=torch.float32).reshape(1, 1, 400, 600)
    y = pad16(x)
    assert y.shape == (1, 1, 400, 608)
    torch.testing.assert_close(y[..., :600], x, rtol=0, atol=0)
    torch.testing.assert_close(y[..., 600:], x[..., 591:599].flip(-1), rtol=0, atol=0)


def test_snr_signal_noise_and_per_image_normalization():
    low = torch.tensor([.1, .5]).reshape(1, 1, 1, 2).repeat(2, 3, 1, 1)
    smooth = torch.tensor([.3, .3]).reshape(1, 1, 1, 2).repeat(2, 3, 1, 1)
    low[1] = 0
    smooth[1] = 0
    result = snr_mask(low, smooth)
    ratio = .3 / (.2 + .0001)
    torch.testing.assert_close(result[0], torch.full((1, 1, 2), ratio / (ratio + .0001)))
    assert torch.equal(result[1], torch.zeros_like(result[1]))


def test_cli_only_low_and_model_binding_and_output():
    assert {a.dest for a in parser()._actions} == {'help', 'low', 'checkpoint', 'config', 'source', 'out'}
