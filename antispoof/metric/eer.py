from antispoof.base.base_metric import BaseMetric
import numpy as np


class EER(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super(EER, self).__init__(name)

    def compute_det_curve(self, target_scores, nontarget_scores):
        n_scores = target_scores.size(0) + nontarget_scores.size(0)
        all_scores = np.concatenate((target_scores, nontarget_scores))
        labels = np.concatenate(
            (np.ones(target_scores.size(0)), np.zeros(nontarget_scores.size(0))))

        # Sort labels based on scores
        indices = np.argsort(all_scores, kind='mergesort')
        labels = labels[indices]

        # Compute false rejection and false acceptance rates
        tar_trial_sums = np.cumsum(labels)
        nontarget_trial_sums = nontarget_scores.size(0) - \
            (np.arange(1, n_scores + 1) - tar_trial_sums)

        # false rejection rates
        frr = np.concatenate(
            (np.atleast_1d(0), tar_trial_sums / target_scores.size(0)))
        far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                            nontarget_scores.size(0)))  # false acceptance rates
        # Thresholds are the sorted scores
        thresholds = np.concatenate(
            (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

        return frr, far, thresholds

    def compute_eer(self, bonafide_scores, other_scores):
        """ 
        Returns equal error rate (EER) and the corresponding threshold.
        """
        bonafide_scores = bonafide_scores.cpu().detach()
        other_scores = other_scores.cpu().detach()
        frr, far, thresholds = self.compute_det_curve(bonafide_scores, other_scores)
        abs_diffs = np.abs(frr - far)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((frr[min_index], far[min_index]))
        return eer, thresholds[min_index]

    def __call__(self, logits, type, **kwargs):
        eer, _ = self.compute_eer(
        bonafide_scores=logits[type == 1, 1],
        other_scores=logits[type== 0, 0])
        return eer
