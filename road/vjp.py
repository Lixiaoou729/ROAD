"""Exact chain-rule staged backward with generator/VAE graphs released in turn."""

import torch


def staged_backward(predict, decode, objective, noise, scale=1.0):
    # Capture RNG before each forward; replay it without consuming extra draws.
    devices = [noise.device.index or 0] if noise.is_cuda else []
    cpu = torch.get_rng_state()
    gpu = torch.cuda.get_rng_state(noise.device) if noise.is_cuda else None
    with torch.no_grad():
        latent = predict(noise)
    cpu_decode = torch.get_rng_state()
    gpu_decode = torch.cuda.get_rng_state(noise.device) if noise.is_cuda else None
    with torch.no_grad():
        image = decode(latent)
    image = image.detach().requires_grad_(True)
    loss = objective(image) * scale
    (image_grad,) = torch.autograd.grad(loss, image)
    value = loss.detach()
    del loss, image
    latent = latent.detach().requires_grad_(True)
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(cpu_decode)
        if gpu_decode is not None:
            torch.cuda.set_rng_state(gpu_decode, noise.device)
        decoded = decode(latent)
        (latent_grad,) = torch.autograd.grad(decoded, latent, image_grad)
    del decoded, image_grad, latent
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(cpu)
        if gpu is not None:
            torch.cuda.set_rng_state(gpu, noise.device)
        predicted = predict(noise)
        predicted.backward(latent_grad)
    return value
