"""Fixed-budget screening using existing ImageCAS-X architectures."""
from __future__ import annotations

import csv
import os
import platform
import time
from functools import lru_cache
from pathlib import Path
import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from models.registry import build_model
from utils.config import BenchmarkConfig
from utils.seeding import seed_everything
from inference import patch_stitch_inference
from .data import TASKS, digest, file_sha256, read_json, select, target_array, training_signature, write_json
from .workstation import worker_init, host_resources
from .prediction import read_prediction, read_run, run_metadata, save_prediction

CLASS_ARGUMENT = {"cas_net": "classes", "ffr_unet": "out_channels", "swin_unetr": "out_channels"}


def build(name, task, params=None):
    if name == "dynunet":
        from monai.networks.nets import DynUNet

        class Wrapped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = DynUNet(3, 1, TASKS[task], kernel_size=[3]*4, strides=[1,2,2,2],
                    upsample_kernel_size=[2]*3, res_block=True, **(params or {}))

            def forward(self, x):
                return {"logits": self.net(x)}
        return Wrapped()
    cfg = BenchmarkConfig({"method_name": name})
    cfg.model.name = name
    cfg.model.params = {**(params or {}), CLASS_ARGUMENT[name]: TASKS[task]}
    return build_model(cfg)


def normalize(image):
    return (np.clip(image, -200, 1000)-400)/600


class Patches(Dataset):
    def __init__(self, rows, task, patch, count, seed, *, training, cache_cases, sampling, hard_negative_root=None, disk_cache=None, augmentation=None):
        if not rows:
            raise ValueError("Empty patient partition.")
        self.rows, self.task = rows, task
        self.patch, self.count, self.seed = np.asarray(patch), count, seed
        self.training, self.sampling = training, sampling
        self.cache_cases, self.hard_negative_root = cache_cases, hard_negative_root
        self.disk_cache = disk_cache
        self.augmentation = augmentation if training else None
        self.spacings = ({case:tuple(nib.affines.voxel_sizes(nib.load(row['image']).affine)) for case,row in rows}
                         if self.augmentation else {})
        self._load = lru_cache(maxsize=cache_cases)(self._cached_case)

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key != "_load"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load = lru_cache(maxsize=self.cache_cases)(self._cached_case)

    def _cached_case(self, index):
        if self.disk_cache is None:
            return self._load_case(index)
        folder = Path(self.disk_cache[self.rows[index][0]])
        return tuple(np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                     for name in ("image", "target", "foreground", "plaque", "healthy", "hard"))

    def _load_case(self, index):
        case, row = self.rows[index]
        reference = nib.load(row["image"])
        image = reference.get_fdata(dtype=np.float32, caching="unchanged")
        target = target_array(row, self.task)
        hard = np.empty((0, 3), int)
        if self.hard_negative_root:
            metadata = read_json(Path(self.hard_negative_root)/"run.json")
            scores, coverage = read_prediction(self.hard_negative_root, case, reference, metadata["task"])
            channel = 2 if metadata["task"] == "multiclass" else 1
            truth = target == (2 if self.task == "multiclass" else 1)
            hard = np.argwhere(coverage & (scores[channel] >= .5) & ~truth)
        pad = [(0, max(0, int(p)-dim)) for p, dim in zip(self.patch, image.shape)]
        if any(v for _, v in pad):
            image, target = np.pad(image, pad, constant_values=-1024), np.pad(target, pad, constant_values=255)
        foreground = np.argwhere((target > 0) & (target < 255))
        plaque = (np.argwhere(target == (2 if self.task == "multiclass" else 1))
                  if self.task in {"multiclass", "plaque"} else np.empty((0,3), int))
        healthy = []
        if self.sampling.get("healthy", 0):
            paths = row.get("branches", {}).values() or [row["label"]]
            for path in paths:
                labels = np.asanyarray(nib.load(path).dataobj)
                if (labels == 1).any() and not (labels == 2).any():
                    healthy.append(np.argwhere(labels == 1))
        healthy = np.concatenate(healthy, axis=0) if healthy else np.empty((0, 3), int)
        return image, target, foreground, plaque, healthy, hard

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        seed, offset = self.seed, 0
        if isinstance(index, tuple):
            index, seed, offset = index
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        row_index=(index + offset) % len(self.rows)
        image, target, foreground, plaque, healthy, hard = self._load(row_index)
        maximum = np.asarray(image.shape)-self.patch
        fractions = [self.sampling.get(name, 0) for name in ("plaque", "healthy", "hard_negative", "foreground")]
        pools = (plaque, healthy, hard, foreground, None)
        candidates = pools[rng.choice(len(pools), p=[*fractions, 1-sum(fractions)])]
        if candidates is not None and len(candidates):
            center = candidates[rng.integers(len(candidates))]
            offset = rng.integers(-self.patch//4, self.patch//4+1) if self.training else 0
            start = np.clip(center+offset-self.patch//2, 0, maximum)
        else:
            start = rng.integers(0, maximum+1)
        window = tuple(slice(int(s), int(s+p)) for s, p in zip(start, self.patch))
        x, y = image[window], target[window]
        flip = self.training and rng.random() < .5
        if self.augmentation:
            from .ctta import augment_hu
            x,y,_ = augment_hu(x,y,self.spacings[self.rows[row_index][0]],self.augmentation,
                               int(rng.integers(0,2**32-1)),axes='xyz')
        x=normalize(x)
        if flip:
            x, y = np.flip(x, 1), np.flip(y, 1)
        return torch.from_numpy(np.ascontiguousarray(x[None])), torch.from_numpy(y.astype(np.int64))


class EpochSampler(Sampler):
    """Send epoch state with each index so persistent spawned workers see it."""
    def __init__(self, count, seed):
        self.count, self.seed, self.offset = count, seed, 0

    def __len__(self):
        return self.count

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        # Match the base-seed draw made by the old per-epoch DataLoader before
        # its RandomSampler draws the patch permutation.
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        order = torch.randperm(self.count, generator=generator)
        return iter((int(i), self.seed, self.offset) for i in order)


def prepare_patch_cache(rows, task, config):
    """Decode once; OS-backed read-only mappings are shared across worker processes.

    Full source hashes prevent reuse after input edits. No image resampling,
    normalization, or annotation changes are performed by this cache.
    """
    if config.get("hard_negative_predictions") or os.environ.get("PLAQUE_DISABLE_DISK_CACHE") == "1":
        return None
    first = rows[0][1]
    root = Path(os.environ.get("PLAQUE_CACHE_DIR") or (Path(first["label"]).resolve().parent.parent / "native_patch_cache"))
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    builder = Patches(rows, task, config["patch"], 1, config["seed"], training=False,
                      cache_cases=0, sampling=config["sampling"])
    def prepare_one(item):
        index,(case,row)=item
        paths = [row["image"], row["branch_lumen"] if task == "branch_lumen" else row["label"]]
        if config["sampling"].get("healthy", 0):
            paths += list(row.get("branches", {}).values()) or [row["label"]]
        sources = {}
        for path in paths:
            path = str(Path(path).resolve())
            sources[path] = file_sha256(path)
        key = digest(dict(schema=1, task=task, patch=config["patch"],
                          healthy=bool(config["sampling"].get("healthy", 0)), sources=sources))
        folder = root / key
        if not (folder / "complete.json").is_file():
            arrays = builder._load_case(index)
            required = sum(a.nbytes for a in arrays)
            if shutil.disk_usage(root).free < required + 256 * 2**20:
                raise OSError(f"Insufficient disk space for patch cache at {root}. Set PLAQUE_CACHE_DIR to a larger SSD.")
            temporary = root / (key + ".tmp-" + uuid.uuid4().hex)
            temporary.mkdir()
            try:
                for name, array in zip(("image", "target", "foreground", "plaque", "healthy", "hard"), arrays):
                    np.save(temporary / f"{name}.npy", array, allow_pickle=False)
                write_json(temporary / "complete.json", dict(key=key, case=case, bytes=required))
                try:
                    temporary.rename(folder)
                except OSError:
                    if not (folder / "complete.json").is_file():
                        raise
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            del arrays
        return case,str(folder)
    workers=int(config.get('cache_build_workers',1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for case,folder in tqdm(pool.map(prepare_one,enumerate(rows)),total=len(rows),
                                desc='Prepare/reuse patch cache',unit='case',ascii=True):
            result[case]=folder
    print(f"Read-only patch cache: {root}", flush=True)
    return result


def objective(logits, target, config):
    valid = target != 255
    labels = target.masked_fill(~valid, 0)
    probs = logits.softmax(dim=1)
    onehot = F.one_hot(labels, logits.shape[1]).movedim(-1, 1)
    dims = (0,2,3,4)
    pred, truth = probs*valid[:,None], onehot*valid[:,None]
    tp = (pred*truth).sum(dims)[1:]
    n_pred, n_true = pred.sum(dims)[1:], truth.sum(dims)[1:]
    mode = config.get("loss", "dice_ce")
    if mode == "focal_tversky":
        score = (tp+1)/(tp+.3*(n_pred-tp)+.7*(n_true-tp)+1)
        return (1-score).pow(1.33).mean()
    weights = config.get("class_weights")
    ce = F.cross_entropy(logits, target, weight=None if weights is None else logits.new_tensor(weights),
                         ignore_index=255, reduction="none")
    if mode == "focal":
        pt = probs.gather(1, labels[:,None])[:,0]
        return ((1-pt).square()*ce).sum()/valid.sum().clamp_min(1)
    ce = ce.sum()/valid.sum().clamp_min(1)
    if mode == "weighted_ce":
        return ce
    base = .5*ce+.5*(1-(2*tp+1)/(n_pred+n_true+1)).mean()
    if mode in {"boundary", "cldice"}:
        from .losses import boundary_dice, cldice
        auxiliary = (boundary_dice(pred[:,1:], truth[:,1:].float()) if mode == "boundary"
                     else cldice(pred[:,1:2], truth[:,1:2].float()))
        return base+config.get("auxiliary_weight", .2)*auxiliary
    if mode != "dice_ce":
        raise ValueError(f"Unknown loss: {mode}")
    return base


def config_for(model, profile, override=None):
    config = read_json(Path(__file__).parent/"configs"/f"{profile}.json")
    config["model"] = model
    if override:
        config.update(read_json(override))
    return config


def _validate_settings(config, task):
    if config.get("hard_negative_predictions") and task not in {"plaque", "multiclass"}:
        raise ValueError("Plaque mining requires plaque targets.")
    if config["sampling"].get("hard_negative", 0) and not config.get("hard_negative_predictions"):
        raise ValueError("Specify training-partition predictions for hard negatives.")
    if config.get("loss") == "cldice" and task not in {"lumen", "multiclass"}:
        raise ValueError("clDice requires a lumen target.")
    patch = np.asarray(config["patch"])
    factor = {"cas_net":16, "ffr_unet":8, "swin_unetr":32, "dynunet":8}[config["model"]]
    if patch.shape != (3,) or (patch < 2*factor).any() or (patch % factor).any():
        raise ValueError(f"Patch dimensions must be >= {2*factor} and divisible by {factor}.")
    for key in ("epochs","steps","batch_size","accumulate","validation_patches","inference_batch"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive.")
    for key in ('cache_build_workers','prefetch_factor'):
        if int(config.get(key,1)) < 1:
            raise ValueError(f'{key} must be positive')
    if config['workers'] < 0:
        raise ValueError('workers must be nonnegative')
    if set(config["sampling"])-{"plaque","healthy","hard_negative","foreground"}:
        raise ValueError("Unknown patch sampling mode.")
    if min(config["sampling"].values(), default=0) < 0 or sum(config["sampling"].values()) > 1:
        raise ValueError("Sampling fractions must be nonnegative and sum to <= 1.")
    if config.get("class_weights") is not None:
        weights = np.asarray(config["class_weights"])
        if weights.shape != (TASKS[task],) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Supply one finite positive class weight per channel.")


def _save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _history_files(out, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (out/"history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in history for k in row)))
        writer.writeheader()
        writer.writerows(history)
    fig, ax = plt.subplots(figsize=(7,4))
    for key in history[0]:
        if key.startswith("val_patch_dice_"):
            ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=key.removeprefix("val_patch_dice_"))
    ax.set(xlabel="Epoch", ylabel="Validation patch Dice", ylim=(0,1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"training_curves.png", dpi=120)
    plt.close(fig)


def train(manifest, task, config, out, *, device=None, resume=False, initialize=None, epoch_callback=None):
    _validate_settings(config, task)
    if config.get('augmentation'):
        from .ctta import settings
        settings(config['augmentation'])
        import ctaug  # Fail before cache preparation if the optional dependency is absent.
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    last = out/"last.pt"
    if last.exists() and not resume:
        raise FileExistsError("Use --resume or select a new output.")
    if resume and initialize:
        raise ValueError("Select resume or new-run initialization.")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    cfg = {**config, "task":task, "split_sha256":manifest["split_sha256"],
           "dataset_sha256":training_signature(manifest)}
    if initialize:
        cfg["initialization_sha256"] = file_sha256(initialize)
    write_json(out/"config.json", cfg, lock=True)
    write_json(out/"environment.json", {"python":platform.python_version(), "torch":str(torch.__version__),
               "device":str(device),"host":host_resources()})
    seed_everything(config["seed"])
    if device.type == "cuda":
        cap = config.get("max_vram_gib",22.)*2**30
        torch.cuda.set_per_process_memory_fraction(min(1.,cap/torch.cuda.get_device_properties(device).total_memory), device)
    model = build(config["model"], task, config.get("model_params")).to(device)
    if initialize:
        state = torch.load(initialize, map_location=device, weights_only=True)
        if any(state["config"][key] != cfg[key] for key in ("model","task","split_sha256","dataset_sha256")):
            raise ValueError("Initialization architecture/task/data mismatch.")
        model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    bf16 = device.type == "cuda" and config["amp"] and torch.cuda.is_bf16_supported()
    names = {"lumen":("lumen",), "plaque":("plaque",), "multiclass":("lumen","plaque"),
             "branch_lumen":("LAD","LCX","RCA")}[task]
    first, history, best, stalled = 0, [], -float("inf"), 0
    if resume:
        state = torch.load(last, map_location=device, weights_only=True)
        if state["config"] != cfg:
            raise ValueError("Resume configuration/data differs from the saved checkpoint.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        first, history, best, stalled = state["epoch"], state["history"], state["best"], state["stalled"]
    training_rows, val_rows = select(manifest,"train"), select(manifest,"val")
    mining = config.get("hard_negative_predictions")
    if mining and read_run(mining, manifest, "train")["task"] not in {"plaque","multiclass"}:
        raise ValueError("Mining run must supply training plaque probabilities.")

    disk_cache = prepare_patch_cache(training_rows + val_rows, task, config) if first < config["epochs"] else None

    def loader(rows, count, seed, training):
        data = Patches(rows, task, config["patch"], count, seed, training=training,
            cache_cases=config["cache_cases"], sampling=config["sampling"],
            hard_negative_root=mining if training else None, disk_cache=disk_cache,
            augmentation=config.get('augmentation'))
        sampler = EpochSampler(count, seed) if training else None
        options = dict(persistent_workers=True, prefetch_factor=int(config.get('prefetch_factor',2))) if config["workers"] > 0 else {}
        return DataLoader(data, batch_size=config["batch_size"], sampler=sampler,
                          num_workers=config["workers"], pin_memory=device.type=="cuda", worker_init_fn=worker_init,
                          generator=torch.Generator().manual_seed(seed), **options)

    val = loader(val_rows, max(config["validation_patches"],len(val_rows)), config["seed"]+10000, False)
    patches = config["steps"]*config["batch_size"]*config["accumulate"]
    batches = loader(training_rows, patches, config["seed"], True)
    for epoch in range(first,config["epochs"]):
        if config["patience"] and stalled >= config["patience"]:
            break
        seed_everything(config["seed"]+epoch+1)
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        offset = epoch*patches % len(training_rows)
        batches.sampler.seed, batches.sampler.offset = config["seed"] + epoch, offset
        model.train()
        optimizer.zero_grad(set_to_none=True)
        training_loss = 0.
        data_wait = compute_wall = 0.
        ready = time.perf_counter()
        progress = tqdm(batches, desc=f"Epoch {epoch+1}/{config['epochs']} train", unit="batch",
                        ascii=True, dynamic_ncols=True, mininterval=2)
        for step,(image,mask) in enumerate(progress,1):
            received = time.perf_counter()
            data_wait += received - ready
            image,mask = image.to(device, non_blocking=True),mask.to(device, non_blocking=True)
            with torch.autocast(device.type,dtype=torch.bfloat16,enabled=bf16):
                logits = model(image)["logits"]
            loss = objective(logits.float(),mask,config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss. Previous completed checkpoints remain usable.")
            (loss/config["accumulate"]).backward()
            if step % config["accumulate"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),config.get("grad_clip",12.),error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            training_loss += loss.item()
            compute_wall += time.perf_counter() - received
            progress.set_postfix(loss=f"{training_loss/step:.4f}", updates=f"{step//config['accumulate']}/{config['steps']}",
                                 wait=f"{data_wait:.1f}s", refresh=False)
            ready = time.perf_counter()
        model.eval()
        tp,denominator,truth = np.zeros(len(names)),np.zeros(len(names)),np.zeros(len(names))
        val_loss = 0.
        with torch.inference_mode():
            for image,mask in tqdm(val, desc=f"Epoch {epoch+1} validation", unit="batch", ascii=True, mininterval=2):
                mask = mask.to(device, non_blocking=True)
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=bf16):
                    logits = model(image.to(device, non_blocking=True))["logits"]
                val_loss += objective(logits.float(),mask,config).item()
                pred = logits.argmax(dim=1).masked_fill(mask == 255,0)
                for c in range(1,len(names)+1):
                    actual,predicted = mask == c,pred == c
                    tp[c-1] += (actual&predicted).sum().item()
                    truth[c-1] += actual.sum().item()
                    denominator[c-1] += actual.sum().item()+predicted.sum().item()
        dices = [float(2*a/b) if g else None for a,b,g in zip(tp,denominator,truth)]
        row = {"epoch":epoch+1,"train_loss":training_loss/len(batches),"val_patch_loss":val_loss/len(val),
               "seconds":time.perf_counter()-start,"optimizer_steps":(epoch+1)*config["steps"],"patches":(epoch+1)*patches,
               "peak_vram_gib":torch.cuda.max_memory_reserved(device)/2**30 if device.type=="cuda" else None}
        row.update(data_wait_seconds=data_wait, train_batch_wall_seconds=compute_wall,
                   train_patches_per_second=patches / max(data_wait + compute_wall, 1e-9))
        row['host_rss_gib']=host_resources().get('process_rss_gib')
        row["gpu_hours_cumulative"] = (sum(r["seconds"] for r in history)+row["seconds"])/3600 if device.type=="cuda" else None
        row.update({f"val_patch_dice_{name}":score for name,score in zip(names,dices)})
        history.append(row)
        scheduler.step()
        defined = [score for score in dices if score is not None]
        score = float(np.mean(defined)) if task == "branch_lumen" and defined else dices[-1]
        value = score if score is not None else -row["val_patch_loss"]
        improved,best = value > best,max(best,value)
        stalled = 0 if improved else stalled+1
        state = {"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                 "config":cfg,"epoch":epoch+1,"history":history,"best":best,"stalled":stalled}
        if improved:
            _save_checkpoint(out/"best.pt",state)
        _save_checkpoint(last,state)
        _history_files(out,history)
        print(row,flush=True)
        if epoch_callback is not None:
            epoch_callback(row)


def infer(manifest, partition, checkpoint, out, *, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    state = torch.load(checkpoint,map_location="cpu",weights_only=True)
    cfg = state["config"]
    task = cfg["task"]
    if cfg["split_sha256"] != manifest["split_sha256"] or cfg["dataset_sha256"] != training_signature(manifest):
        raise ValueError("Checkpoint data/split mismatch.")
    if device.type == "cuda":
        cap = cfg.get("max_vram_gib",22.)*2**30
        torch.cuda.set_per_process_memory_fraction(min(1.,cap/torch.cuda.get_device_properties(device).total_memory),device)
    model = build(cfg["model"],task,cfg.get("model_params")).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    identity = file_sha256(checkpoint)
    run_metadata(out,manifest,partition,task,manifest.get("geometry","native"),model=cfg["model"],
        initialization="fine-tuned" if cfg.get("initialization_sha256") else "random",
        checkpoint=str(Path(checkpoint).resolve()),checkpoint_sha256=identity,candidate_id=identity,
        **({'training_augmentation':cfg['augmentation']} if cfg.get('augmentation') else {}))
    for case,row in select(manifest,partition):
        ref = nib.load(row["image"])
        image = torch.from_numpy(normalize(ref.get_fdata(dtype=np.float32)))[None,None]
        with torch.inference_mode():
            logits = patch_stitch_inference(model,image,cfg["patch"],device,batch_size=cfg["inference_batch"],
                                            overlap=cfg["inference_overlap"])
            probs = logits.softmax(dim=1)[0].numpy()
        save_prediction(out,case,ref,probs)
