#!/usr/bin/env python3
"""Generate COCO-17 keypoint labels for a video with detectron2 Keypoint R-CNN
(top-scoring person per frame). Output kp[N,17,3] float32 (x, y, score) in
pixel coords, zeros where no person is detected.

python teacher/keypoints_gt.py --video A_video.avi --out A_kp.npz
"""
import argparse
import numpy as np, torch, cv2


def main(a):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    pred = DefaultPredictor(cfg)

    cap = cv2.VideoCapture(a.video)
    N = int(cap.get(7))
    end = a.end if a.end > 0 else N
    out = np.zeros((end - a.start, 17, 3), np.float32)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    for i in range(a.start, end):
        ok, frame = cap.read()
        if not ok:
            break
        with torch.no_grad():
            inst = pred(frame)["instances"]
        inst = inst[inst.pred_classes == 0]
        if len(inst) > 0:
            k = int(inst.scores.argmax())
            out[i - a.start] = inst.pred_keypoints[k].cpu().numpy()
        if (i + 1) % 500 == 0:
            print(i + 1, "/", end, flush=True)
    np.savez(a.out, kp=out, start=a.start, end=end)
    det = float((out[:, :, 2].max(1) > 0).mean())
    print(f"saved {a.out}  person detected in {100 * det:.0f}% of frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    main(ap.parse_args())
