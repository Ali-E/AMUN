import torch
import torch.nn as nn
from attack import Attack


class PGDL2(Attack):
    
    def __init__(
        self,
        model,
        eps=0.04,
        # alpha=0.2,
        steps=50,
        random_start=False,
        eps_for_division=1e-10,
    ):
        super().__init__("PGDL2", model)
        self.eps = eps
        self.alpha = self.eps / 10. # alpha
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self.supported_mode = ["default", "targeted"]

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        """
        Returns:
        -------
        adv_images : torch.Tensor  # shape = images.shape
        orig_pred  : torch.Tensor  # model prediction on the clean inputs
        final_pred : torch.Tensor  # prediction on the adversarial inputs
        eps_tensor : torch.Tensor  # shape (N,) – smallest ε per sample
        delta_norm : torch.Tensor  # L2 norm of the final perturbation
        """
        device          = images.device
        batch_size_all  = images.size(0)
        batch_size      = images.size(0)

        eps_all         = torch.full((batch_size_all,), self.eps,  device=device)
        alpha_all       = eps_all / 10.0
        adv_images_all  = images.clone().detach()
        adv_preds_all   =  torch.zeros_like(labels) - 1 # labels.clone().detach()
        indices_all     = torch.arange(batch_size_all, device=device)
        indices         = indices_all.clone()
        delta_all       = torch.zeros_like(eps_all)
        eps             = torch.full((batch_size,), self.eps,  device=device)
        alpha           = eps / 10.0

        images          = images.clone().detach()
        adv_images      = images.clone().detach()
        labels          = labels.clone().detach()
        orig_pred       = None
        loss_fn         = nn.CrossEntropyLoss()

        cont_flag   = True
        while indices.numel() > 0:
            for _ in range(self.steps):
                adv_images.requires_grad_()
                logits      = self.get_logits(adv_images)
                preds       = logits.argmax(1)

                still_correct   = preds.eq(labels)
                if still_correct.sum() == 0:
                    delta = adv_images - images[indices]
                    d_norm = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
                    adv_images_all[indices] = adv_images
                    eps_all[indices]        = eps
                    alpha_all[indices]      = alpha 
                    adv_preds_all[indices]      = preds

                    cont_flag = False
                    break  
                else:
                    if ~still_correct.sum() != 0:
                        adv_preds_all[indices[~still_correct]] = preds[~still_correct]

                if orig_pred is None:
                    orig_pred = preds.detach()

                loss        = loss_fn(logits, labels)
                grad        = torch.autograd.grad(loss, adv_images)[0]

                grad_norm   = grad.view(batch_size, -1).norm(p=2, dim=1) \
                              + self.eps_for_division
                grad        = grad / grad_norm.view(-1, 1, 1, 1)
                grad[~still_correct] = 0.0

                adv_images  = adv_images.detach() + \
                              (alpha.view(-1, 1, 1, 1) * grad)

                delta       = adv_images - images[indices]
                d_norm      = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
                factor      = torch.minimum(eps / d_norm, torch.ones_like(d_norm))
                adv_images  = torch.clamp(images[indices] + delta *
                                          factor.view(-1, 1, 1, 1),
                                          min=0.0, max=1.0).detach()

                adv_images    = adv_images[still_correct] #.requires_grad_()
                batch_size    = still_correct.sum().item()
                eps           = eps[still_correct]
                alpha         = alpha[still_correct]
                preds         = preds[still_correct]
                labels        = labels[still_correct]
                indices       = indices[still_correct]
                delta_all[indices] = d_norm[still_correct]

                adv_images_all[indices] = adv_images
                eps_all[indices]        = eps
                alpha_all[indices]      = alpha 
                # preds_all[indices]      = preds


            if not cont_flag:
                break

            torch.cuda.empty_cache()

            with torch.no_grad():
                preds         = self.get_logits(adv_images).argmax(1)
            still_correct   = preds.eq(labels)

            if still_correct.sum() != 0:
                eps[still_correct]   *= 2
                alpha            = eps / 10.0

        # final stats
        delta_final = (adv_images_all - images).view(batch_size_all, -1)
        delta_norm  = delta_final.norm(p=2, dim=1)

        return adv_images_all, orig_pred, adv_preds_all, eps_all, delta_norm
