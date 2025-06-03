import torch
import numpy as np

def aggregate_subsamples(all_video_ids, all_logits):
    # Aggregate logits (NOT softmax outputs)
    video_logits_dict = {}
    for video_id, logit in zip(all_video_ids, all_logits):
        if video_id not in video_logits_dict:
            video_logits_dict[video_id] = []
        video_logits_dict[video_id].append(logit)

    aggregated_logits = []
    aggregated_video_ids = []
    for video_id, logits in video_logits_dict.items():
        logits = np.stack(logits, axis=0)  # (num_subsamples, num_classes)
        avg_logits = np.mean(logits, axis=0)  # Average over logits
        aggregated_logits.append(avg_logits)
        aggregated_video_ids.append(video_id)

    aggregated_logits = np.stack(aggregated_logits, axis=0)
    aggregated_video_ids = np.array(aggregated_video_ids)

    # Apply softmax after aggregation
    aggregated_probs = torch.softmax(torch.from_numpy(aggregated_logits), dim=1).numpy()

    return aggregated_video_ids, aggregated_probs