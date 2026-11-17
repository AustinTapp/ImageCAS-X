"""CTTA_Aug adapters for raw-HU native crops and normalized nnU-Net crops."""
from contextlib import contextmanager
from copy import deepcopy
import random
import numpy as np

PINNED_CTTA_COMMIT = '7fbebab4d391f45432f92e8374adaa2770d986eb'
DEFAULT = dict(step=0.10, motion=0.0, step_motion=0.10, metal=0.0, wire=0.0,
               calcification=0.0, step_weight=[0.05,0.20], motion_range=[0.01,0.025],
               severity=[0.10,0.30], allow_synthetic_objects=False)
KINDS = ('step','motion','step_motion','metal','wire','calcification')


def settings(config):
    cfg = {**DEFAULT, **deepcopy(config or {})}
    if set(cfg) - set(DEFAULT):
        raise ValueError('Unknown CTTA settings: '+str(set(cfg)-set(DEFAULT)))
    if any(not np.isfinite(cfg[k]) or not 0 <= cfg[k] <= 1 for k in KINDS):
        raise ValueError('CTTA probabilities must be finite values in [0,1]')
    for key in ('step_weight','motion_range','severity'):
        lo,hi=cfg[key]
        if not 0 <= lo <= hi <= 1:
            raise ValueError(f'Invalid CTTA range {key}')
    if any(cfg[k] for k in ('metal','wire','calcification')) and not cfg['allow_synthetic_objects']:
        raise ValueError('Object insertion requires allow_synthetic_objects=true: labels represent underlying anatomy, not newly inserted disease/devices.')
    return cfg


@contextmanager
def seeded(seed):
    # CTAug uses Python and NumPy globals. Keep callers' random streams intact.
    py, np_state = random.getstate(), np.random.get_state()
    random.seed(int(seed)); np.random.seed(int(seed) % 2**32)
    try:
        yield
    finally:
        random.setstate(py); np.random.set_state(np_state)


def augment_hu(image, target, spacing, config, seed, *, axes='xyz', ignore=255):
    """Augment one raw-HU patch; no validation or inference caller uses this.

    Projection-based insertion is opt-in corruption of the appearance. It does
    not create new disease labels. Calcification anchors only on existing plaque.
    Motion uses the upstream implementation with zero fill and no cut/resize;
    newly exposed voxels are ignored, rather than labelled as background.
    """
    cfg=settings(config)
    from ctaug import StepTransform, MetalTransform, WireTransform, CalcificationTransform
    from ctaug.step.functional import MotionAugmentation, StepMotionAugmentation, motion_core
    x=np.array(image,dtype=np.float32,copy=True)
    y=np.array(target,copy=True)
    if x.ndim!=3 or y.shape!=x.shape or axes not in {'xyz','zyx'}:
        raise ValueError('CTTA expects matching scalar 3D arrays and xyz/zyx axes')
    if axes=='xyz':
        x=x.transpose(2,1,0).copy(); y=y.transpose(2,1,0).copy(); spacing=tuple(spacing)[::-1]
    if len(spacing)!=3 or min(spacing)<=0:
        raise ValueError('CTTA requires positive per-axis spacing in mm')
    valid=(y!=ignore) & (y>=0) & (y!=255)
    y=np.where(valid,y,0)
    info=[]
    with seeded(seed):
        for kind in KINDS:
            if random.random() >= cfg[kind]:
                continue
            if kind=='step':
                out=StepTransform(cutoff_index=0,cut_off_pixel_value_weight=cfg['step_weight'],
                                  mean_pixel_value=250.,normalize=False)(image=x,segmentation=y)
                x=out['image']; info.append(dict(kind=kind,parameters=out['StepTransform_info']))
            elif kind in ('motion','step_motion'):
                cls=MotionAugmentation if kind=='motion' else StepMotionAugmentation
                kw=dict(motion_move_range=cfg['motion_range'],normalize=False)
                if kind=='step_motion':
                    kw.update(cut_off_pixel_value_weight=cfg['step_weight'],mean_pixel_value=250.)
                x,y,params=cls(**kw)(x,y,cropped_value='zero',cutoff_index=0)
                valid=motion_core(valid.astype(np.uint8),params['cutoff_index'],params['move_index'],
                    'zero',params['cut_off_position'],valid.shape,0,params['move_value'],
                    move_left=params['move_left'],input_type='label').astype(bool)
                x=np.where(valid,x,-1024.).astype(np.float32)
                info.append(dict(kind=kind,parameters=params))
            else:
                labels=(2,) if kind=='calcification' else (0,)
                anchors=np.where(valid,y,-1)
                if not np.isin(anchors,labels).any():
                    continue
                cls={'metal':MetalTransform,'wire':WireTransform,'calcification':CalcificationTransform}[kind]
                out=cls(spacing=tuple(spacing),include_labels=labels,exclude_labels=None,
                        max_n_specs=1,severity=cfg['severity'],verbose=False)(image=x,segmentation=anchors)
                x=out['image']; info.append(dict(kind=kind,parameters=out[cls.__name__+'_info']))
    y=np.where(valid,np.rint(y),ignore).astype(target.dtype)
    x=np.asarray(x,dtype=np.float32)
    if not np.isfinite(x).all() or not np.isin(y,np.unique(np.append(target,ignore))).all():
        raise ValueError('CTTA produced nonfinite intensities or unexpected class IDs')
    if axes=='xyz':
        x=x.transpose(2,1,0); y=y.transpose(2,1,0)
    return np.ascontiguousarray(x),np.ascontiguousarray(y),info


class NNUNetCTTA:
    """Prepend CTAug to nnU-Net 2.8 training transforms, before deep supervision."""
    def __init__(self, wrapped, config, spacing, mean, std):
        self.wrapped,self.config,self.spacing=wrapped,settings(config),spacing
        self.mean,self.std=float(mean),float(std)
        if self.std<=0:
            raise ValueError('Invalid nnU-Net CT normalization standard deviation')

    def __call__(self, **data):
        import torch
        image,seg=data['image'],data['segmentation']
        if image.shape[0]!=1 or seg.shape[0]!=1 or image.device.type!='cpu':
            raise ValueError('CTTA nnU-Net adapter requires one CT channel and one scalar CPU segmentation')
        hu=image[0].numpy()*self.std+self.mean
        label=seg[0].numpy()
        seed=int(np.random.randint(0,2**32-1))
        x,y,_=augment_hu(hu,label,self.spacing,self.config,seed,axes='zyx',ignore=-1)
        data['image']=torch.from_numpy(((x-self.mean)/self.std)[None]).to(image.dtype)
        data['segmentation']=torch.from_numpy(y[None]).to(seg.dtype)
        return self.wrapped(**data)


def main():
    """Export one original/augmented training crop for visual review."""
    import argparse,json
    from pathlib import Path
    import nibabel as nib
    from .data import read_json,target_array,save_image
    p=argparse.ArgumentParser(description=main.__doc__)
    p.add_argument('--dataset',required=True); p.add_argument('--config',required=True)
    p.add_argument('--case',required=True); p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=2028); p.add_argument('--patch',type=int,default=64)
    args=p.parse_args()
    row=read_json(args.dataset)['cases'][args.case]
    if row['partition']!='train':
        raise ValueError('Preview is restricted to training cases')
    ref=nib.load(row['image']);label=target_array(row,'multiclass')
    foreground=np.argwhere((label>0)&(label<255))
    center=foreground[len(foreground)//2] if len(foreground) else np.asarray(label.shape)//2
    size=np.minimum(args.patch,np.asarray(label.shape));start=np.clip(center-size//2,0,np.asarray(label.shape)-size)
    window=tuple(slice(int(a),int(a+b)) for a,b in zip(start,size))
    x=ref.get_fdata(dtype=np.float32)[window];y=label[window]
    a,b,info=augment_hu(x,y,nib.affines.voxel_sizes(ref.affine),read_json(args.config),args.seed)
    affine=ref.affine.copy();affine[:3,3]=nib.affines.apply_affine(ref.affine,start)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    for name,arr in [('original_ct',x),('original_label',y),('augmented_ct',a),('augmented_label',b)]:
        save_image(out/(name+'.nii.gz'),arr,affine)
    (out/'augmentation_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(f'Preview: {out}; applied {len(info)} transforms. Another seed may sample a different set, including none.')


if __name__=='__main__':
    main()
