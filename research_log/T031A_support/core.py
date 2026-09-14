"""T031-A feature-only nearest-source score. No image/reference/metric inputs."""
import math
import torch
def distances(source_features,state_features,x_mean,x_scale):
    source=(source_features.double()-x_mean.double())/x_scale.double()
    states=(state_features.double()-x_mean.double())/x_scale.double()
    return torch.cat([torch.cdist(x,source,p=2,compute_mode='donot_use_mm_for_euclid_dist').amin(1)/math.sqrt(28)
                      for x in states.split(128)])
