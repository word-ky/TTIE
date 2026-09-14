def forward(input_, model_restoration):
    b, c, h, w = input_.shape
    H, W = ((h + factor) // factor * factor, (w + factor) // factor * factor)
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    if h < 3000 and w < 3000:
        if args.self_ensemble:
            restored = self_ensemble(input_, model_restoration)
        else:
            restored = model_restoration(input_)
    else:
        input_1 = input_[:, :, :, 1::2]
        input_2 = input_[:, :, :, 0::2]
        if args.self_ensemble:
            restored_1 = self_ensemble(input_1, model_restoration)
            restored_2 = self_ensemble(input_2, model_restoration)
        else:
            restored_1 = model_restoration(input_1)
            restored_2 = model_restoration(input_2)
        restored = torch.zeros_like(input_)
        restored[:, :, :, 1::2] = restored_1
        restored[:, :, :, 0::2] = restored_2
    restored = restored[:, :, :h, :w]
    restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    return restored
