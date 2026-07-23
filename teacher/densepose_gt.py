#!/usr/bin/env python3
"""Generate detectron2 DensePose 24-part I-maps for a single-subject video (top-
scoring person per frame). Output dp[end-start,360,640] uint8 (0=bg,1..24)."""
import argparse, numpy as np, torch, cv2
def main(a):
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from densepose import add_densepose_config
    from densepose.vis.extractor import DensePoseResultExtractor
    cfg=get_cfg(); add_densepose_config(cfg); cfg.merge_from_file(a.config); cfg.MODEL.WEIGHTS=a.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.5
    pred=DefaultPredictor(cfg); ext=DensePoseResultExtractor()
    cap=cv2.VideoCapture(a.video); N=int(cap.get(7)); end=a.end if a.end>0 else N
    OH,OW=360,640; out=np.zeros((end-a.start,OH,OW),np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES,a.start); i=a.start
    while i<end:
        ok,frame=cap.read()
        if not ok: break
        H,W=frame.shape[:2]; sx,sy=OW/W,OH/H
        with torch.no_grad(): inst=pred(frame)["instances"]
        inst=inst[inst.pred_classes==0]
        if len(inst)>0:
            k=int(inst.scores.argmax()); res=ext(inst[k])
            dpr=res[0][0] if isinstance(res[0],list) else res[0]
            ilab=dpr.labels.cpu().numpy().astype(np.uint8)
            box=inst[k].pred_boxes.tensor.cpu().numpy()[0]
            ox0,oy0,ox1,oy1=int(box[0]*sx),int(box[1]*sy),int(box[2]*sx),int(box[3]*sy)
            ox0,oy0=max(0,ox0),max(0,oy0); ox1,oy1=min(OW,ox1),min(OH,oy1)
            if ox1>ox0 and oy1>oy0:
                out[i-a.start,oy0:oy1,ox0:ox1]=cv2.resize(ilab,(ox1-ox0,oy1-oy0),interpolation=cv2.INTER_NEAREST)
        i+=1
        if i%500==0: print(i,"/",end,flush=True)
    np.savez(a.out,dp=out,start=a.start,end=end); print("saved",a.out,"fg",round(float((out>0).mean()),3),flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--start",type=int,default=0); ap.add_argument("--end",type=int,default=-1)
    ap.add_argument("--config",default="detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
    ap.add_argument("--weights",default="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl")
    main(ap.parse_args())
