"""REFERENCE_GRADIENT_DIAGNOSTIC_ONLY. No optimizer, selector or deployable consumer."""
import torch
def reference_gradient(model,low,reference):
    output=model(low)
    loss=(output.double()-reference.double()).square().mean()
    gradient,=torch.autograd.grad(loss,model.raw)
    return gradient
