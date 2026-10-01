"""Synthetic-probe-only PromptIR skip-tensor spill schedule for native 4K.

Copies the official PromptIR.forward operation order and weights, moving
encoder skip tensors to CPU until their corresponding decoder concat. It is
not accepted for target execution until full-model equivalence and native-size
synthetic output are independently checked.
"""

import torch


def schedule_skip_spill(model, threshold=1_000_000):
    assert model.decoder
    original = model.forward

    def forward(inp_img, noise_emb=None):
        if inp_img.shape[-2] * inp_img.shape[-1] < threshold:
            return original(inp_img, noise_emb)
        inp_enc_level1 = model.patch_embed(inp_img)
        out_enc_level1 = model.encoder_level1(inp_enc_level1)
        skip1 = out_enc_level1.cpu()
        inp_enc_level2 = model.down1_2(out_enc_level1)
        del inp_enc_level1, out_enc_level1
        out_enc_level2 = model.encoder_level2(inp_enc_level2)
        skip2 = out_enc_level2.cpu()
        inp_enc_level3 = model.down2_3(out_enc_level2)
        del inp_enc_level2, out_enc_level2
        out_enc_level3 = model.encoder_level3(inp_enc_level3)
        skip3 = out_enc_level3.cpu()
        inp_enc_level4 = model.down3_4(out_enc_level3)
        del inp_enc_level3, out_enc_level3
        latent = model.latent(inp_enc_level4)
        del inp_enc_level4
        dec3_param = model.prompt3(latent)
        latent = torch.cat([latent, dec3_param], 1)
        del dec3_param
        latent = model.noise_level3(latent)
        latent = model.reduce_noise_level3(latent)
        inp_dec_level3 = model.up4_3(latent)
        del latent
        inp_dec_level3 = torch.cat([inp_dec_level3, skip3.to(inp_img.device)], 1)
        del skip3
        inp_dec_level3 = model.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = model.decoder_level3(inp_dec_level3)
        del inp_dec_level3
        dec2_param = model.prompt2(out_dec_level3)
        out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
        del dec2_param
        out_dec_level3 = model.noise_level2(out_dec_level3)
        out_dec_level3 = model.reduce_noise_level2(out_dec_level3)
        inp_dec_level2 = model.up3_2(out_dec_level3)
        del out_dec_level3
        inp_dec_level2 = torch.cat([inp_dec_level2, skip2.to(inp_img.device)], 1)
        del skip2
        inp_dec_level2 = model.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = model.decoder_level2(inp_dec_level2)
        del inp_dec_level2
        dec1_param = model.prompt1(out_dec_level2)
        out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
        del dec1_param
        out_dec_level2 = model.noise_level1(out_dec_level2)
        out_dec_level2 = model.reduce_noise_level1(out_dec_level2)
        inp_dec_level1 = model.up2_1(out_dec_level2)
        del out_dec_level2
        inp_dec_level1 = torch.cat([inp_dec_level1, skip1.to(inp_img.device)], 1)
        del skip1
        out_dec_level1 = model.decoder_level1(inp_dec_level1)
        del inp_dec_level1
        out_dec_level1 = model.refinement(out_dec_level1)
        return model.output(out_dec_level1) + inp_img

    model.forward = forward
