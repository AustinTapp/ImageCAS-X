"""Coverage-aware probability fusion, specialist combination and held-out gating."""
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.special import expit, logit

from .data import C26, digest, read_json, select, target_array, write_json
from .prediction import read_prediction, read_run, run_metadata, save_prediction
from .protection import guard


def _score(probs, task):
    if task == "plaque":
        return probs[1]
    if task == "multiclass":
        return probs[2]
    raise ValueError("Plaque fusion requires plaque or multiclass probabilities.")


def expert_identity(metadata):
    return {key:metadata.get(key) for key in ("model","checkpoint_sha256","candidate_id","task","geometry")}


def _aligned_scores(manifest, partition, roots, case, reference):
    scores,supports = [],[]
    for root in roots:
        metadata = read_run(root,manifest,partition)
        probs,coverage = read_prediction(root,case,reference,metadata["task"])
        scores.append(_score(probs,metadata["task"]))
        supports.append(coverage)
    return np.stack(scores),np.stack(supports)


def fuse_arrays(scores, supports, *, method="soft", weights=None, coefficients=None):
    if scores.shape != supports.shape or scores.ndim < 2 or not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1:
        raise ValueError("Expected aligned finite probabilities and coverage.")
    count = len(scores)
    weight = np.ones(count) if weights is None else np.asarray(weights,float)
    if weight.shape != (count,) or not np.isfinite(weight).all() or (weight < 0).any() or not weight.any():
        raise ValueError("Supply one nonnegative weight per expert and a positive total.")
    weight = weight.reshape((count,)+(1,)*(scores.ndim-1))*supports
    values = scores
    if method == "vote":
        values = (scores >= .5).astype(float)
    elif method == "confidence":
        p = np.clip(scores,1e-5,1-1e-5)
        confidence = 1+(p*np.log(p)+(1-p)*np.log(1-p))/np.log(2)
        weight *= np.maximum(.05,confidence)
    elif method == "rule":
        weight *= 1+4*np.std(scores,axis=0)[None]*np.abs(scores-.5)*2
    elif method == "learned":
        if coefficients is None or len(coefficients["coef"]) != count:
            raise ValueError("Fitted gate coefficient count differs.")
        score = expit(np.tensordot(coefficients["coef"],logit(np.clip(scores,1e-5,1-1e-5)),axes=(0,0))+coefficients["intercept"])
        coverage = supports.all(axis=0)
        return np.where(coverage,score,0).astype(np.float32),coverage
    elif method != "soft":
        raise ValueError(f"Unknown fusion method: {method}")
    norm = weight.sum(axis=0)
    coverage = norm > 0
    total = np.sum(values*weight,axis=0)
    np.divide(total,norm,out=total,where=coverage)
    return total.astype(np.float32),coverage


def _geometry(metadata):
    labels = [row["geometry"] for row in metadata]
    if any("oracle" in label for label in labels):
        return "oracle-assisted"
    return "fully automatic" if all(label in {"native","fully automatic"} for label in labels) else "upstream-provenance-unverified"


def ensemble(manifest, partition, roots, out, *, method="soft", weights=None, gating=None, threshold=.5, frozen=None):
    if len(roots) < 2 or len(set(str(Path(root).resolve()) for root in roots)) != len(roots):
        raise ValueError("Supply at least two distinct experts.")
    metadata = [read_run(root,manifest,partition) for root in roots]
    gate = read_json(gating) if gating else None
    if method == "learned":
        if gate is None or gate["split_sha256"] != manifest["split_sha256"] or gate["expert_ids"] != [expert_identity(row) for row in metadata]:
            raise ValueError("Gate split or expert identity/order mismatch.")
    params = {"method":method,"weights":weights,"threshold":threshold,
              "experts":[expert_identity(row) for row in metadata],"gate":gate}
    guard(manifest,partition,frozen,digest(params))
    run_metadata(out,manifest,partition,"plaque",_geometry(metadata),model=f"ensemble_{method}",parameters=params,
                 calibration_ids=[] if gate is None else gate["calibration_ids"],candidate_id=digest(params))
    import time
    started = time.perf_counter()
    extraction = {}
    for case,row in select(manifest,partition):
        ref = nib.load(row["image"])
        scores,supports = _aligned_scores(manifest,partition,roots,case,ref)
        score,coverage = fuse_arrays(scores,supports,method=method,weights=weights,coefficients=gate)
        save_prediction(out,case,ref,np.stack([np.where(coverage,1-score,0),score]),coverage,
                        seg=(coverage & (score >= threshold)).astype(np.uint8))
        extraction[case] = {}
        for branch in ("LAD","LCX","RCA"):
            available = []
            for root in roots:
                failure = Path(root)/"extraction.json"
                available.append(read_json(failure)[case][branch]["success"] if failure.is_file() else True)
            extraction[case][branch] = {"success":all(available) if method == "learned" else any(available)}
    write_json(Path(out)/"extraction.json",extraction)
    write_json(Path(out)/"timing.json",{"fusion_seconds":time.perf_counter()-started})


def fit_gate(manifest, roots, out, *, seed=2028, per_case=12000):
    from sklearn.linear_model import LogisticRegression
    metadata = [read_run(root,manifest,"val") for root in roots]
    rows = select(manifest,"val")
    if len(rows) < 4 or len(roots) < 2:
        raise ValueError("Gating requires two experts and four validation patients.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    calibration = [rows[i] for i in order[:len(rows)//2]]
    assessment = sorted(rows[i][0] for i in order[len(rows)//2:])
    features,targets = [],[]
    for case,row in calibration:
        ref = nib.load(row["image"])
        scores,supports = _aligned_scores(manifest,"val",roots,case,ref)
        truth = target_array(row,"plaque")
        eligible = np.flatnonzero(supports.all(axis=0))
        if not len(eligible):
            continue
        indices = rng.choice(eligible,min(per_case,len(eligible)),replace=False)
        features.append(logit(np.clip(scores.reshape(len(roots),-1)[:,indices].T,1e-5,1-1e-5)))
        targets.append(truth.ravel()[indices])
    if not targets or np.unique(np.concatenate(targets)).size < 2:
        raise ValueError("Calibration samples must contain plaque and non-plaque.")
    classifier = LogisticRegression(C=1.,max_iter=200,random_state=seed).fit(np.concatenate(features),np.concatenate(targets))
    gate = {"coef":classifier.coef_[0].tolist(),"intercept":float(classifier.intercept_[0]),"seed":seed,
        "split_sha256":manifest["split_sha256"],"expert_ids":[expert_identity(row) for row in metadata],
        "calibration_ids":sorted(case for case,_ in calibration),"assessment_ids":assessment,
        "supervision":"Held-out validation calibration. Every calibration patient is excluded from assessment."}
    write_json(out,gate,lock=True)
    return gate


def refine(manifest, partition, root, out, *, min_volume_mm3=0., threshold=.5, lumen_root=None,
           max_lumen_distance_mm=None, hu_min=None, hu_max=None, frozen=None):
    metadata = read_run(root,manifest,partition)
    params = {"min_volume_mm3":min_volume_mm3,"threshold":threshold,"max_lumen_distance_mm":max_lumen_distance_mm,
              "hu_min":hu_min,"hu_max":hu_max,"upstream":expert_identity(metadata)}
    if min_volume_mm3 < 0 or not 0 < threshold < 1:
        raise ValueError("Invalid minimum volume or threshold.")
    if max_lumen_distance_mm is not None and (not lumen_root or max_lumen_distance_mm <= 0):
        raise ValueError("Lumen-distance refinement needs a lumen run and positive distance.")
    if lumen_root:
        lumen_meta = read_run(lumen_root,manifest,partition)
        if lumen_meta["task"] not in {"lumen","multiclass","branch_lumen"}:
            raise ValueError("Independent lumen probabilities are required.")
        params["lumen_expert"] = expert_identity(lumen_meta)
    guard(manifest,partition,frozen,digest(params))
    geometry = "oracle-assisted" if lumen_root and "oracle" in lumen_meta["geometry"] else metadata["geometry"]
    run_metadata(out,manifest,partition,"plaque",geometry,model="postprocessed",parameters=params,
                 candidate_id=digest(params),calibration_ids=metadata.get("calibration_ids",[]))
    for case,row in select(manifest,partition):
        ref = nib.load(row["image"])
        probs,coverage = read_prediction(root,case,ref,metadata["task"])
        score = _score(probs,metadata["task"]).copy()
        keep = coverage.copy()
        if lumen_root and max_lumen_distance_mm is not None:
            lp,lc = read_prediction(lumen_root,case,ref,lumen_meta["task"])
            labels = lp.argmax(axis=0)
            lumen = ((labels > 0) if lumen_meta["task"] == "branch_lumen" else labels == 1) & lc
            if not lumen.any():
                keep[:] = False
            else:
                keep &= ndi.distance_transform_edt(~lumen,sampling=nib.affines.voxel_sizes(ref.affine)) <= max_lumen_distance_mm
        if hu_min is not None or hu_max is not None:
            ct = ref.get_fdata(dtype=np.float32)
            if hu_min is not None:
                keep &= ct >= hu_min
            if hu_max is not None:
                keep &= ct <= hu_max
        prediction = keep & (score >= threshold)
        components,_ = ndi.label(prediction,C26)
        sizes = np.bincount(components.ravel())*np.prod(nib.affines.voxel_sizes(ref.affine))
        remove = sizes < min_volume_mm3
        remove[0] = False
        prediction &= ~remove[components]
        save_prediction(out,case,ref,np.stack([np.where(coverage,1-score,0),score]),coverage,seg=prediction.astype(np.uint8))
    failure = Path(root)/"extraction.json"
    if failure.is_file():
        write_json(Path(out)/"extraction.json",read_json(failure))


def specialists(manifest, partition, lumen_root, plaque_root, out, frozen=None):
    lm,pm = read_run(lumen_root,manifest,partition),read_run(plaque_root,manifest,partition)
    if lm["task"] not in {"lumen","multiclass","branch_lumen"} or pm["task"] not in {"plaque","multiclass"}:
        raise ValueError("Lumen and plaque specialist runs are required.")
    params = {"lumen":expert_identity(lm),"plaque":expert_identity(pm),
              "rule":"P(plaque)=p, P(lumen)=(1-p)*l, P(background)=(1-p)*(1-l)"}
    guard(manifest,partition,frozen,digest(params))
    run_metadata(out,manifest,partition,"multiclass",_geometry([lm,pm]),model="independent_specialists",
                 parameters=params,candidate_id=digest(params))
    for case,row in select(manifest,partition):
        ref = nib.load(row["image"])
        lumen,ls = read_prediction(lumen_root,case,ref,lm["task"])
        plaque,ps = read_prediction(plaque_root,case,ref,pm["task"])
        coverage = ls & ps
        l = lumen[1:].sum(axis=0) if lm["task"] == "branch_lumen" else lumen[1]
        p = _score(plaque,pm["task"])
        probs = np.stack([(1-p)*(1-l),(1-p)*l,p])*coverage[None]
        save_prediction(out,case,ref,probs,coverage)
    failure = Path(plaque_root)/"extraction.json"
    if failure.is_file():
        write_json(Path(out)/"extraction.json",read_json(failure))
