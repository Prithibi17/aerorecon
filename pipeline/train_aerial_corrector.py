"""Train a small residual depth corrector on TartanAir ground truth."""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from moge.model.v2 import MoGeModel


def read_depth(path):
    rgba=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if rgba is None: raise ValueError(f'Cannot read {path}')
    return np.squeeze(rgba.view('<f4'),axis=-1)


class Corrector(nn.Module):
    """Small residual network; predicts a bounded log-depth shape correction."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(4,32,5,2,2),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.SiLU(),
            nn.Conv2d(64,96,3,2,1),nn.SiLU(),
            nn.ConvTranspose2d(96,64,4,2,1),nn.SiLU(),
            nn.ConvTranspose2d(64,32,4,2,1),nn.SiLU(),
            nn.ConvTranspose2d(32,1,4,2,1),nn.Tanh())

    def forward(self,x): return self.net(x)*.7


class CacheDataset(Dataset):
    def __init__(self,files): self.files=files
    def __len__(self): return len(self.files)
    def __getitem__(self,index):
        data=np.load(self.files[index])
        return torch.from_numpy(data['features']),torch.from_numpy(data['target']),torch.from_numpy(data['valid'])


def pairs(root):
    images=sorted(root.glob('P*/image_lcam_front/*.png'))
    result=[]
    for image in images:
        depth=image.parents[1]/'depth_lcam_front'/image.name.replace('_lcam_front','_lcam_front_depth')
        if depth.exists(): result.append((image,depth,image.parents[1].name))
    return result


def normalized_log(depth,valid):
    log=np.log(np.maximum(depth,1e-4));median=np.median(log[valid])
    return np.clip(log-median,-4,4),median


def prepare(dataset_root,cache,limit,device):
    all_pairs=pairs(dataset_root)
    if not all_pairs: raise ValueError('No paired RGB/depth images found')
    # Even sampling preserves every trajectory while bounding laptop runtime.
    chosen=[all_pairs[i] for i in np.linspace(0,len(all_pairs)-1,min(limit,len(all_pairs))).astype(int)]
    model=MoGeModel.from_pretrained(str(Path('models/MoGe-2-Base/model.pt'))).to(device).eval()
    cache.mkdir(parents=True,exist_ok=True)
    records=[]
    with torch.inference_mode():
        for number,(image_path,depth_path,trajectory) in enumerate(chosen):
            rgb=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
            rgb=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA)
            truth=cv2.resize(read_depth(depth_path),(256,256),interpolation=cv2.INTER_NEAREST)
            valid=np.isfinite(truth)&(truth>.1)&(truth<500)
            if valid.mean()<.25: continue
            image=torch.tensor(rgb/255,dtype=torch.float32,device=device).permute(2,0,1)
            output=model.infer(image,resolution_level=1,fov_x=90)
            predicted=output['depth'].cpu().numpy()
            valid &= output['mask'].cpu().numpy().astype(bool)&np.isfinite(predicted)&(predicted>0)
            pred_log,_=normalized_log(predicted,valid);truth_log,_=normalized_log(truth,valid)
            target=np.clip(truth_log-pred_log,-.7,.7)[None].astype(np.float32)
            features=np.concatenate([(rgb.transpose(2,0,1)/255).astype(np.float32),pred_log[None].astype(np.float32)])
            destination=cache/f'{trajectory}_{number:05d}.npz'
            np.savez_compressed(destination,features=features,target=target,valid=valid[None].astype(np.float32))
            records.append((destination,trajectory))
            if (number+1)%100==0: print(f'Cached {number+1}/{len(chosen)} ground-truth samples',flush=True)
    return records


def error(model,loader,device):
    model.eval();base=[];corrected=[]
    with torch.inference_mode():
        for x,target,valid in loader:
            x,target,valid=x.to(device),target.to(device),valid.to(device)
            prediction=model(x)
            base.extend(torch.abs(torch.exp(-target)-1)[valid.bool()].cpu().tolist())
            corrected.extend(torch.abs(torch.exp(prediction-target)-1)[valid.bool()].cpu().tolist())
    return float(np.mean(base)),float(np.mean(corrected))


def run(args):
    torch.manual_seed(17);np.random.seed(17);random.seed(17)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    records=prepare(Path(args.dataset),Path(args.cache),args.samples,device)
    trajectories=sorted(set(t for _,t in records))
    if len(trajectories)<2: raise ValueError('Need at least two trajectories for a leakage-safe split')
    validation={trajectories[-1]}
    train=[p for p,t in records if t not in validation];val=[p for p,t in records if t in validation]
    train_loader=DataLoader(CacheDataset(train),batch_size=args.batch,shuffle=True,num_workers=0)
    val_loader=DataLoader(CacheDataset(val),batch_size=args.batch,num_workers=0)
    model=Corrector().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    best=None;history=[]
    for epoch in range(args.epochs):
        model.train()
        for x,target,valid in train_loader:
            x,target,valid=x.to(device),target.to(device),valid.to(device)
            prediction=model(x);loss=(torch.abs(prediction-target)*valid).sum()/valid.sum().clamp_min(1)
            optimizer.zero_grad();loss.backward();optimizer.step()
        baseline,result=error(model,val_loader,device);history.append(dict(epoch=epoch+1,baseline_abs_rel=baseline,corrected_abs_rel=result))
        print(f'Epoch {epoch+1}: validation {baseline:.4f} -> {result:.4f}',flush=True)
        if best is None or result<best[0]:best=(result,{k:v.detach().cpu() for k,v in model.state_dict().items()})
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    improved=best[0]<history[0]['baseline_abs_rel']*.98
    torch.save(dict(state_dict=best[1],architecture='aerial-depth-corrector-v1',input_size=256,dataset='TartanAir V2 ArchVizTinyHouseDay',validation_trajectories=sorted(validation)),output/'model.pt')
    metrics=dict(train_samples=len(train),validation_samples=len(val),trajectories=len(trajectories),validation_trajectories=sorted(validation),history=history,best_corrected_abs_rel=best[0],baseline_abs_rel=history[0]['baseline_abs_rel'],promoted=improved,license='TartanAir V2 CC BY 4.0')
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (output/'README.md').write_text(f'# Aerial depth corrector\n\nFrozen MoGe-2 predictions were corrected using {len(train)} TartanAir training images. {len(val)} images from whole held-out trajectories were used only for validation. Baseline normalized-depth AbsRel: {metrics["baseline_abs_rel"]:.4f}; corrected: {best[0]:.4f}. Promoted: {improved}. Synthetic validation does not prove improvement on real drone footage. Dataset: TartanAir V2, CC BY 4.0.\n',encoding='utf-8')
    if not improved:(output/'DO_NOT_USE').write_text('Validation did not improve enough.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);parser.add_argument('--cache',required=True);parser.add_argument('--out',required=True);parser.add_argument('--samples',type=int,default=800);parser.add_argument('--epochs',type=int,default=8);parser.add_argument('--batch',type=int,default=8)
    run(parser.parse_args())
