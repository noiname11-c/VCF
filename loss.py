import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import log_normal_dist


class Orthogonal_Loss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(Orthogonal_Loss, self).__init__()
        self.epsilon = epsilon
    
    def compute_orthogonal_loss(self, embeddings):
        B, L, C = embeddings.shape
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        cosine_sim_matrix = torch.einsum('blc,bkc->blk', embeddings_norm, embeddings_norm)
        cosine_sim_squared = cosine_sim_matrix ** 2
        eye_mask = torch.eye(L, device=embeddings.device).unsqueeze(0)
        cosine_sim_squared = cosine_sim_squared * (1 - eye_mask)
        cosine_loss = cosine_sim_squared.mean()
        return cosine_loss
    def forward(self, embeddings, args):
        Loss_noraml_text = self.compute_orthogonal_loss(embeddings[:, 0:args.prompt_num,:])
        Loss_abnormal_text = self.compute_orthogonal_loss(embeddings[:,args.prompt_num:,:])
        orthogonal_loss = Loss_noraml_text + Loss_abnormal_text
        return orthogonal_loss

class FocalLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(FocalLoss, self).__init__()
        self.epsilon = epsilon
    def forward(self, pred, gt, gamma=2.0, alpha=1, mask_ratio=1.0):
        if gt.dim() == 4 and gt.size(1) == 1:
            gt = gt.squeeze(1)
        elif gt.dim() == 2:
            gt = gt.unsqueeze(0)
        if gt.dim() != 3:
            raise ValueError(f"FocalLoss expected gt with shape (B,H,W) or (H,W) or (B,1,H,W), but got {tuple(gt.shape)}")

        gt_one_hot = F.one_hot(gt.long(), num_classes=2).permute(0, 3, 1, 2).float()
        pt = (pred * gt_one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** gamma  # (1 - p_t)^γ
        focal_loss = -alpha * focal_weight * torch.log(pt + self.epsilon)
        fg_mask = gt > 0
        bg_mask = ~fg_mask
        fg_loss = focal_loss * fg_mask.float()
        bg_loss = focal_loss * bg_mask.float()
        fg_pixels = fg_mask.sum().float()
        bg_pixels = bg_mask.sum().float()
        fg_loss_final = fg_loss.sum() / (fg_pixels + self.epsilon)
        bg_loss_final = bg_loss.sum() / (bg_pixels + self.epsilon)
        loss = fg_loss_final  + bg_loss_final
        return loss


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
    
    def compute_loss(self, pred, target):
        target_sum = torch.sum(target)
        intersection = torch.sum(pred * target) 
        union = torch.sum(pred) + target_sum 
        dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
        loss = 1 - dice
        return loss 

    def forward(self, pred, target):
 
        target = target.float()  
        loss_f = self.compute_loss(pred[:, 1, :, :],  target.clone())
        loss_b = self.compute_loss(pred[:, 0, :, :],  1 - target)
        #loss = loss_f + loss_b
        loss = loss_f
        return loss

class SoftF1Loss(nn.Module):
    """
    Differentiable soft-F1 loss for image-level (sample-level) binary classification.
    Computes a smooth approximation of the F1 score via softmax probabilities,
    so the gradient directly guides the model toward higher f1_sp.
    Works even under severe class imbalance (no hard threshold needed).
    """
    def __init__(self, epsilon: float = 1e-8):
        super(SoftF1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits  : [B, 2]  raw scores for [normal, anomaly]
        # targets : [B]     0 = normal, 1 = anomaly
        p = F.softmax(logits, dim=-1)[:, 1]   # anomaly probability  [B]
        y = targets.float()
        tp = (p * y).sum()
        fp = (p * (1.0 - y)).sum()
        fn = ((1.0 - p) * y).sum()
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn + self.epsilon)
        return 1.0 - f1


class SoftFbetaLoss(nn.Module):
    """
    Differentiable soft-Fbeta loss for image-level binary classification.
    beta > 1 puts more emphasis on recall, which often improves f1_sp
    on anomaly-detection datasets with minority positives.
    """
    def __init__(self, beta: float = 1.5, epsilon: float = 1e-8):
        super(SoftFbetaLoss, self).__init__()
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=-1)[:, 1]
        y = targets.float()
        tp = (p * y).sum()
        fp = (p * (1.0 - y)).sum()
        fn = ((1.0 - p) * y).sum()
        b2 = self.beta * self.beta
        fbeta = ((1.0 + b2) * tp) / ((1.0 + b2) * tp + b2 * fn + fp + self.epsilon)
        return 1.0 - fbeta


class DynamicBalancedBCELoss(nn.Module):
    """
    Dynamic class-balanced BCE for image-level classification.
    Uses score = logit_anomaly - logit_normal and computes pos_weight from
    the current mini-batch to reduce class-imbalance sensitivity.
    """
    def __init__(self, max_pos_weight: float = 20.0, epsilon: float = 1e-8):
        super(DynamicBalancedBCELoss, self).__init__()
        self.max_pos_weight = max_pos_weight
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, 2], targets: [B] in {0,1}
        y = targets.float()
        score = logits[:, 1] - logits[:, 0]
        pos = y.sum()
        neg = y.numel() - pos
        pos_weight = (neg + self.epsilon) / (pos + self.epsilon)
        pos_weight = torch.clamp(pos_weight, min=1.0, max=self.max_pos_weight)
        return F.binary_cross_entropy_with_logits(score, y, pos_weight=pos_weight)


class PairwiseLogisticLoss(nn.Module):
    """
    Pairwise logistic ranking loss for AUROC/AP optimization.
    Minimizes log(1 + exp(-(s_pos - s_neg) / tau)).
    """
    def __init__(self, tau: float = 0.2):
        super(PairwiseLogisticLoss, self).__init__()
        self.tau = tau

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scores = F.softmax(logits, dim=-1)[:, 1]
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            return scores.sum() * 0.0
        pair_gap = (pos.unsqueeze(1) - neg.unsqueeze(0)) / self.tau
        return F.softplus(-pair_gap).mean()


class CompAUCLoss(nn.Module):
    """
    Compositional squared-loss surrogate for AUROC / AP (image-level).
    Based on: Yuan et al. "Compositional Training for End-to-End Deep AUC
    Maximization" (ICLR 2021).

    For every (anomaly, normal) pair the optimal gap is 1,
    so minimising E[(1 - (s+ - s-))**2] directly maximises AUROC.
    Gradients flow through both positive and negative branch;
    no .detach() is used.
    """
    def __init__(self, epsilon: float = 1e-8):
        super(CompAUCLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits : [B, 2]   labels : [B]
        scores = F.softmax(logits, dim=-1)[:, 1]   # [B]  — full gradient path
        pos = scores[labels == 1]   # [P]
        neg = scores[labels == 0]   # [N]
        if pos.numel() == 0 or neg.numel() == 0:
            return scores.sum() * 0.0
        # squared loss: want (pos - neg) == 1
        diff = 1.0 - (pos.unsqueeze(1) - neg.unsqueeze(0))   # [P, N]
        return (diff ** 2).mean()


def binary_loss_function(x_recon, x, z_mu, z_var, z_0, z_k, log_det_jacobians, z_size, cuda, beta=1, summ = True, log_vamp_zk = None, if_rec = True):
    """
    z_mu: mean of z_0
    z_var: variance of z_0
    z_0: first stochastic latent variable
    z_k: last stochastic latent variable
    log_det_jacobians: log det jacobian
    beta: beta for annealing according to Equation 20
    log_vamp_zk: default None but log_p(zk) if VampPrior used
    the function returns: Free Energy Bound (ELBO), reconstruction loss, kl
    """
    batch_size = x.size(0) 
    
    logvar=torch.zeros(batch_size, z_size)    
    if cuda == True:                      
        logvar=logvar.cuda()
        
    # calculate log_p(zk) under standard Gaussian unless log_p(zk) under VampPrior given
    if log_vamp_zk is None:
        log_p_zk = log_normal_dist(z_k, mean=0, logvar=logvar, dim=1) # ln p(z_k) = N(0,I)
    else:
        log_p_zk = log_vamp_zk
    
                 
    log_q_z0 = log_normal_dist(z_0, mean=z_mu, logvar=z_var.log(), dim=1)

    log_p_zk = log_p_zk + 1e-8
    log_q_z0 = log_q_z0 + 1e-8

    
    if (summ == True):  ## Computes the binary loss function with summing over batch dimension 
        
        #Reconstruction loss: Binary cross entropy
        reconstruction_loss = nn.MSELoss(reduction='sum')

        if if_rec:
            log_p_xz = reconstruction_loss(x_recon, x)  #-log_p(x|z_k)
        else:
            log_p_xz = 0
        log_p_xz = log_p_xz
        kl = torch.sum(log_q_z0 - log_p_zk) - torch.sum(log_det_jacobians) #sum over batches
        #elbo = elbo / batch_size
        log_p_xz = log_p_xz / batch_size
        kl = kl / batch_size

        elbo = 0
        
        return elbo, log_p_xz, kl
    
    else:              ## Computes the binary loss function without summing over batch dimension (used during testing) 
        if len(log_det_jacobians.size()) > 1:
            log_det_jacobians = log_det_jacobians.view(log_det_jacobians.size(0), -1).sum(-1)

        reconstruction_loss = nn.BCELoss(reduction='none')
        log_p_xz = reconstruction_loss(x_recon.view(batch_size, -1), x.view(batch_size, -1))  #-log_p(x|z_k)
        log_p_xz = torch.sum(log_p_xz, dim=1)
        
        #Equation (20)
        elbo = log_q_z0 - log_p_zk - log_det_jacobians + log_p_xz 

        return elbo, log_p_xz, (log_q_z0 - log_p_zk - log_det_jacobians)
