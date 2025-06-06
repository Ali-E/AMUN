import torch
import torch.nn as nn

from attack import Attack

class PGDL2(Attack):
    
    def __init__(
        self,
        model,
        eps=0.01,
        # alpha=0.2,
        steps=10,
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
        Vector-ised L2 PGD that finds an individual ε for every image.

        Returns
        -------
        adv_images : torch.Tensor  # shape = images.shape
        orig_pred  : torch.Tensor  # model prediction on the clean inputs
        final_pred : torch.Tensor  # prediction on the adversarial inputs
        eps_tensor : torch.Tensor  # shape (N,) – smallest ε per sample
        delta_norm : torch.Tensor  # L2 norm of the final perturbation
        """
        device      = images.device
        batch_size  = images.size(0)

        # one ε and α per sample
        eps         = torch.full((batch_size,), self.eps,  device=device)
        alpha       = eps / 10.0

        images      = images.clone().detach()
        adv_images  = images.clone().detach()
        labels      = labels.clone().detach()

        finished    = torch.zeros(batch_size, dtype=torch.bool, device=device)
        orig_pred   = None
        loss_fn     = nn.CrossEntropyLoss()

        while not finished.all():
            for _ in range(self.steps):
                adv_images.requires_grad_()
                logits      = self.get_logits(adv_images)
                preds       = logits.argmax(1)

                still_correct   = preds.eq(labels)
                if still_correct.sum() == 0:
                    # if all samples are misclassified, we can stop
                    break  

                if orig_pred is None:
                    orig_pred = preds.detach()

                loss        = loss_fn(logits, labels)
                grad        = torch.autograd.grad(loss, adv_images)[0]

                print(adv_images.shape, logits.shape, labels.shape, eps.shape, alpha.shape)

                # normalise gradient sample-wise
                grad_norm   = grad.view(batch_size, -1).norm(p=2, dim=1) \
                              + self.eps_for_division
                grad        = grad / grad_norm.view(-1, 1, 1, 1)
                # make the grad 0 where still correct is false:
                grad[~still_correct] = 0.0

                adv_images  = adv_images.detach() + \
                              (alpha.view(-1, 1, 1, 1) * grad)

                # project back to the L2 ball of radius ε (per sample)
                delta       = adv_images - images
                d_norm      = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
                # d_norm      = delta.view(batch_size, -1).norm(p=2, dim=1)
                factor      = torch.minimum(eps / d_norm, torch.ones_like(d_norm))
                adv_images  = torch.clamp(images + delta *
                                          factor.view(-1, 1, 1, 1),
                                          min=0.0, max=1.0).detach()

            # check which images are still correct
            with torch.no_grad():
                preds         = self.get_logits(adv_images).argmax(1)
            still_correct   = preds.eq(labels)
            finished        |= ~still_correct

            # double ε (and α) **only** for unfinished samples
            if (~finished).any():
                eps[~finished]   *= 2
                alpha            = eps / 10.0

        # final stats
        delta_final = (adv_images - images).view(batch_size, -1)
        delta_norm  = delta_final.norm(p=2, dim=1)

        return adv_images, orig_pred, preds, eps, delta_norm
