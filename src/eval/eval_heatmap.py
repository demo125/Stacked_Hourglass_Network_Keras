from heatmap_process import post_process_heatmap
import data_process
import numpy as np
import copy


def get_predicted_kp_from_htmap(heatmap, meta, outres):
    # nms to get location
    kplst = post_process_heatmap(heatmap)
    kps = np.array(kplst)

    # use meta information to transform back to original image
    mkps = copy.copy(kps)
    for i in range(kps.shape[0]):
        mkps[i, 0:2] = data_process.transform(kps[i], meta['center'], meta['scale'], res=outres, invert=1, rot=0)

    return mkps


def cal_kp_distance(pre_kp, gt_kp, norm, threshold):
    if gt_kp[0] > 1 and gt_kp[1] > 1:
        print('gt - pred', gt_kp[0:2], pre_kp[0:2])
        dif = np.linalg.norm(gt_kp[0:2] - pre_kp[0:2]) / norm
        if dif < threshold:
            # good prediction
            return 1, dif
        else:  # failed
            return 0, dif
    else:
        return -1, -1


def heatmap_accuracy(predhmap, meta, norm, threshold):
    pred_kps = post_process_heatmap(predhmap)
    pred_kps = np.array(pred_kps)

    gt_kps = meta['tpts'] #TODO wtf is this

    good_pred_count = 0
    failed_pred_count = 0
    avg_dif = []
    for i in range(gt_kps.shape[0]):
        dis, dif = cal_kp_distance(pred_kps[i, :], gt_kps[i, :], norm, threshold)
        if dis == 0:
            failed_pred_count += 1
            avg_dif.append(dif)
        elif dis == 1:
            good_pred_count += 1
            avg_dif.append(dif)
    m = np.mean(np.array(avg_dif))
    print("AVG DIF", m)
    return good_pred_count, failed_pred_count, m 


def cal_heatmap_acc(prehmap, metainfo, threshold):
    sum_good, sum_fail = 0, 0
    #TODO wtf norm ?
    
    ms = []
    if len(prehmap.shape) > 3:
        for i in range(prehmap.shape[0]):
            _prehmap = prehmap[i, :, :, :]
            good, bad, m = heatmap_accuracy(_prehmap, metainfo[i], norm=6.4, threshold=threshold)
            ms.append(m)
            sum_good += good
            sum_fail += bad
    else:
        
        good, bad = heatmap_accuracy(prehmap, metainfo[0], norm=6.4, threshold=threshold)
        
        sum_good += good
        sum_fail += bad

    return sum_good, sum_fail, ms
