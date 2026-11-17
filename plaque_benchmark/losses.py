"""Optional differentiable morphological losses."""
import torch
import torch.nn.functional as F


def soft_erode(image):
    kernels = ((3, 1, 1), (1, 3, 1), (1, 1, 3))
    return torch.stack([-F.max_pool3d(-image, kernel, stride=1, padding=tuple(k//2 for k in kernel))
                        for kernel in kernels]).amin(dim=0)


def soft_skeleton(image, iterations=10):
    eroded = soft_erode(image)
    skeleton = F.relu(image-F.max_pool3d(eroded, 3, stride=1, padding=1))
    for _ in range(iterations):
        image = eroded
        eroded = soft_erode(image)
        delta = F.relu(image-F.max_pool3d(eroded, 3, stride=1, padding=1))
        skeleton = skeleton+F.relu(delta-skeleton*delta)
    return skeleton


def cldice(probability, target, iterations=10):
    sp, st = soft_skeleton(probability, iterations), soft_skeleton(target, iterations)
    precision = ((sp*target).sum()+1e-6)/(sp.sum()+1e-6)
    recall = ((st*probability).sum()+1e-6)/(st.sum()+1e-6)
    return 1-2*precision*recall/(precision+recall).clamp_min(1e-6)


def boundary_dice(probability, target):
    p = F.max_pool3d(probability, 3, stride=1, padding=1)-soft_erode(probability)
    g = F.max_pool3d(target, 3, stride=1, padding=1)-soft_erode(target)
    return 1-(2*(p*g).sum()+1)/(p.sum()+g.sum()+1)
