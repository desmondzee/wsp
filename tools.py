import torch


def structure(w):
    return "\n".join(f"{k:<60}{str(tuple(v.shape)):<24}{v.numel()}" for k, v in w.items())


def mat(t):
    return t.reshape(t.shape[0], -1) if t.dim() > 1 else t.reshape(-1, 1)


def stats(t):
    t = t.float()
    s = torch.linalg.svdvals(mat(t))
    p = s / s.sum()
    return {
        "shape": tuple(t.shape),
        "norm": t.norm().item(),
        "mean": t.mean().item(),
        "std": t.std().item() if t.numel() > 1 else 0.0,
        "absmax": t.abs().max().item(),
        "frac_zero": (t == 0).float().mean().item(),
        "effective_rank": torch.exp(-(p * torch.log(p + 1e-12)).sum()).item(),
    }


def svd(t, k=8):
    u, s, vh = torch.linalg.svd(mat(t.float()), full_matrices=False)
    return u[:, :k], s[:k], vh[:k]


def path(*ts):
    out = mat(ts[0].float())
    for t in ts[1:]:
        out = mat(t.float()) @ out
    return out


def compare(w, ref):
    out = {}
    for k, v in w.items():
        if k in ref and torch.is_floating_point(v):
            a, b = v.float().flatten(), ref[k].float().flatten()
            out[k] = {
                "rel_change": ((a - b).norm() / (b.norm() + 1e-12)).item(),
                "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
            }
    return out


def diff(a, b):
    changed, total, per = 0, 0, {}
    for k, v in a.items():
        d = b[k].float() - v.float()
        n = int((d != 0).sum())
        total += v.numel()
        if n:
            changed += n
            per[k] = {"changed": n, "rel_l2": (d.norm() / (v.float().norm() + 1e-12)).item()}
    return {"changed": changed, "total": total, "fraction": changed / max(total, 1), "tensors": per}


def project_out(t, v, side="out"):
    m = mat(t.float())
    v = v.float() / v.float().norm()
    m = m - torch.outer(v, v @ m) if side == "out" else m - torch.outer(m @ v, v)
    return m.reshape(t.shape).to(t.dtype)


def add_rank1(t, u, v, scale=1.0):
    return (mat(t.float()) + scale * torch.outer(u.float(), v.float())).reshape(t.shape).to(t.dtype)
