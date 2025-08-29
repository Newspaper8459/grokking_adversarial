import logging
import time
from contextlib import contextmanager
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from schema.config import Config


class PGD(object):
  """Untargeted PGD attack. Modified from torchattack.attack to allow non-[0,1] domain data

  https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/pgd.html#PGD
  """
  def __init__(
    self,
    model: nn.Module,
    config: Config,
    random_start: bool = True,
  ):
    self.model = model
    self.eps = config.adversarial.atk_eps
    self.alpha = config.adversarial.atk_alpha
    self.steps = config.adversarial.atk_epochs
    self.random_start = random_start
    self.device = config.device
    self.dmax = config.adversarial.dmax
    self.dmin = config.adversarial.dmin

  def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    ## TODO: add targeted attacks

    images = images.clone().detach().to(self.device)
    labels = labels.clone().detach().to(self.device)

    loss = torch.nn.CrossEntropyLoss()
    adv_images = images.clone().detach()
    dmin = images.min() if self.dmin is None else self.dmin
    dmax = images.max() if self.dmax is None else self.dmax

    if self.random_start:
      # Starting at a uniformly random point
      adv_images = adv_images + torch.rand_like(adv_images)*2*self.eps - self.eps
      adv_images = torch.clamp(adv_images, min=dmin, max=dmax).detach()

    for _ in range(self.steps):
      adv_images.requires_grad = True
      outputs = self.model(adv_images)

      cost = loss(outputs, labels)

      # Update adversarial images
      grad = torch.autograd.grad(
          cost, adv_images, retain_graph=False, create_graph=False
      )[0]

      adv_images = adv_images.detach() + self.alpha * grad.sign()
      delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
      adv_images = torch.clamp(images + delta, min=dmin, max=dmax).detach()

    return adv_images


class TPGD(object):
    r"""PGD based on KL-Divergence loss in the paper 'Theoretically Principled Trade-off between Robustness and Accuracy'

    [https://arxiv.org/abs/1901.08573]

    Distance Measure : Linf

    Arguments:
      model (nn.Module): model to attack.
      eps (float): strength of the attack or maximum perturbation. (Default: 8/255)
      alpha (float): step size. (Default: 2/255)
      steps (int): number of steps. (Default: 10)

    Shape:
      - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
      - output: :math:`(N, C, H, W)`.

    Examples::
      >>> attack = torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=10)
      >>> adv_images = attack(images)

    """

    def __init__(
      self,
      model: nn.Module,
      config: Config,
    ):
        self.eps = config.adversarial.atk_eps
        self.alpha = config.adversarial.atk_alpha
        self.steps = config.adversarial.atk_epochs
        self.supported_mode = ["default"]
        self.model = model
        self.dmin = config.adversarial.dmin
        self.dmax = config.adversarial.dmax
        self.device = config.device

    def __call__(self, images: torch.Tensor, labels: torch.Tensor|None=None):
      images = images.clone().detach().to(self.device)
      logit_ori = self.model(images).detach()

      adv_images = images + 0.001 * torch.randn_like(images)
      adv_images = torch.clamp(adv_images, min=self.dmin, max=self.dmax).detach()

      loss = torch.nn.KLDivLoss(reduction="sum")

      for _ in range(self.steps):
        adv_images.requires_grad = True
        logit_adv = self.model(adv_images)

        # Calculate loss
        cost = loss(F.log_softmax(logit_adv, dim=1), F.softmax(logit_ori, dim=1))

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )[0]

        adv_images = adv_images.detach() + self.alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + delta, min=self.dmin, max=self.dmax).detach()

      return adv_images

class APGD(object):
    r"""APGD in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'

    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2

    Arguments:
      model (nn.Module): model to attack.
      norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
      eps (float): maximum perturbation. (Default: 8/255)
      steps (int): number of steps. (Default: 10)
      n_restarts (int): number of random restarts. (Default: 1)
      seed (int): random seed for the starting point. (Default: 0)
      loss (str): loss function optimized. ['ce', 'dlr'] (Default: 'ce')
      eot_iter (int): number of iteration for EOT. (Default: 1)
      rho (float): parameter for step-size update (Default: 0.75)
      verbose (bool): print progress. (Default: False)

    Shape:
      - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
      - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
      - output: :math:`(N, C, H, W)`.

    Examples::
      >>> attack = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
      >>> adv_images = attack(images, labels)

    """

    def __init__(
      self,
      model: nn.Module,
      config: Config,
      norm: Literal['Linf', 'L2'] = "Linf",
      n_restarts: int = 1,
      eot_iter: int = 1,
      rho: float = 0.75,
      verbose: bool = False,
    ):
      self.eps = config.adversarial.atk_eps
      self.steps = config.adversarial.atk_epochs
      if norm not in ['Linf', 'L2']:
        raise NotImplementedError

      self.norm = norm
      self.n_restarts = n_restarts
      self.seed = config.seed
      self.loss = config.train.loss
      self.eot_iter = eot_iter
      self.thr_decr = rho
      self.verbose = verbose
      self.supported_mode = ["default"]
      self.model = model
      self.dmin = config.adversarial.dmin
      self.dmax = config.adversarial.dmax
      self.device = config.device
      self.logger = logging.getLogger('__main__')

    def __call__(self, images: torch.Tensor, labels: torch.Tensor):
      images = images.clone().detach().to(self.device)
      labels = labels.clone().detach().to(self.device)
      _, adv_images = self.perturb(images, labels, cheap=True)

      return adv_images

    def check_oscillation(
      self,
      x: np.ndarray,
      j: int,
      k: int,
      y5: Any,
      k3: float = 0.75
    ) -> np.ndarray:
      t = np.zeros(x.shape[1])
      for counter5 in range(k):
        t += x[j - counter5] > x[j - counter5 - 1]

      return t <= k * k3 * np.ones(t.shape)

    def check_shape(self, x: np.ndarray) -> np.ndarray:
      return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
      x_sorted, ind_sorted = x.sort(dim=1)
      ind = (ind_sorted[:, -1] == y).float()

      return -(
        x[np.arange(x.shape[0]), y]
        - x_sorted[:, -2] * ind
        - x_sorted[:, -1] * (1.0 - ind)
      ) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def attack_single_run(
      self,
      x_in: torch.Tensor,
      y_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
      x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
      y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
      logger = logging.getLogger('__main__')

      self.steps_2, self.steps_min, self.size_decr = (
        max(int(0.22 * self.steps), 1),
        max(int(0.06 * self.steps), 1),
        max(int(0.03 * self.steps), 1),
      )
      if self.verbose:
        logger.info(
          "parameters: ", self.steps, self.steps_2, self.steps_min, self.size_decr
        )

      if self.norm == "Linf":
        t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
        x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
          self.device
        ).detach() * t / (
          t.reshape([t.shape[0], -1])
          .abs()
          .max(dim=1, keepdim=True)[0]
          .reshape([-1, 1, 1, 1])
        )
      elif self.norm == "L2":
        t = torch.randn(x.shape).to(self.device).detach()
        x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
          self.device
        ).detach() * t / (
          (t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
        )
      else:
        raise NotImplementedError

      x_adv = x_adv.clamp(self.dmin, self.dmax)
      x_best = x_adv.clone()
      x_best_adv = x_adv.clone()
      loss_steps = torch.zeros([self.steps, x.shape[0]])
      loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]])
      acc_steps = torch.zeros_like(loss_best_steps)

      if self.loss == "ce":
        criterion_indiv = nn.CrossEntropyLoss(reduction="none")
      elif self.loss == "dlr":
        criterion_indiv = self.dlr_loss
      else:
        raise NotImplementedError("unknown loss")

      x_adv.requires_grad_()
      grad = torch.zeros_like(x)
      for _ in range(self.eot_iter):
        with torch.enable_grad():
          # 1 forward pass (eot_iter = 1)
          logits = self.model(x_adv)
          loss_indiv = criterion_indiv(logits, y)
          loss = loss_indiv.sum()

          # 1 backward pass (eot_iter = 1)
        grad += torch.autograd.grad(loss, [x_adv])[0].detach()

      grad /= float(self.eot_iter)
      grad_best = grad.clone()

      acc = cast(torch.Tensor, logits).detach().max(1)[1] == y
      acc_steps[0] = acc + 0
      loss_best = cast(torch.Tensor, loss_indiv).detach().clone()

      step_size = (
        self.eps
        * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach()
        * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
      )
      x_adv_old = x_adv.clone()
      k = self.steps_2 + 0
      u = np.arange(x.shape[0])
      counter3 = 0

      loss_best_last_check = loss_best.clone()
      reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

      # n_reduced = 0
      for i in range(self.steps):
        # gradient step
        with torch.no_grad():
          x_adv = x_adv.detach()
          grad2 = x_adv - x_adv_old
          x_adv_old = x_adv.clone()

          a = 0.75 if i > 0 else 1.0

          if self.norm == "Linf":
            x_adv_1 = x_adv + step_size * torch.sign(grad)
            x_adv_1 = torch.clamp(
              torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps),
              self.dmin,
              self.dmax,
            )
            x_adv_1 = torch.clamp(
              torch.min(
                torch.max(
                  x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                  x - self.eps,
                ),
                x + self.eps,
              ),
              self.dmin,
              self.dmax,
            )

          elif self.norm == "L2":
            x_adv_1 = x_adv + step_size * grad / (
              (grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
            )
            x_adv_1 = torch.clamp(
              x
              + (x_adv_1 - x)
              / (
                ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                + 1e-12
              )
              * torch.min(
                self.eps * torch.ones(x.shape).to(self.device).detach(),
                ((x_adv_1 - x) ** 2)
                .sum(dim=(1, 2, 3), keepdim=True)
                .sqrt(),
              ),
              self.dmin,
              self.dmax,
            )
            x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
            x_adv_1 = torch.clamp(
              x
              + (x_adv_1 - x)
              / (
                ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                + 1e-12
              )
              * torch.min(
                self.eps * torch.ones(x.shape).to(self.device).detach(),
                ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                + 1e-12,
              ),
              self.dmin,
              self.dmax,
            )
          else:
            raise NotImplementedError

          x_adv = x_adv_1 + 0.0

        # get gradient
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
          with torch.enable_grad():
            # 1 forward pass (eot_iter = 1)
            logits = self.model(x_adv)
            loss_indiv = criterion_indiv(logits, y)
            loss = loss_indiv.sum()

          # 1 backward pass (eot_iter = 1)
          grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)

        pred = cast(torch.Tensor, logits).detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        x_best_adv[(pred == 0).nonzero().squeeze()] = (
          x_adv[(pred == 0).nonzero().squeeze()] + 0.0
        )
        if self.verbose:
          self.logger.info(f"iteration: {i} - Best loss: {loss_best.sum():.6f}")

        # check step size
        with torch.no_grad():
          y1 = loss_indiv.detach().clone()
          loss_steps[i] = y1.cpu() + 0
          ind = (y1 > loss_best).nonzero().squeeze()
          x_best[ind] = x_adv[ind].clone()
          grad_best[ind] = grad[ind].clone()
          loss_best[ind] = y1[ind] + 0
          loss_best_steps[i + 1] = loss_best + 0

          counter3 += 1

          if counter3 == k:
            fl_oscillation = self.check_oscillation(
              loss_steps.detach().cpu().numpy(),
              i,
              k,
              loss_best.detach().cpu().numpy(),
              k3=self.thr_decr,
            )
            fl_reduce_no_impr = (~reduced_last_check) * (
              loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy()
            )  # nopep8
            fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
            reduced_last_check = np.copy(fl_oscillation)
            loss_best_last_check = loss_best.clone()

            if np.sum(fl_oscillation) > 0:
              step_size[u[fl_oscillation]] /= 2.0
              n_reduced = fl_oscillation.astype(float).sum()

              fl_oscillation = np.where(fl_oscillation)

              x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
              grad[fl_oscillation] = grad_best[fl_oscillation].clone()

            counter3 = 0
            k = np.maximum(k - self.size_decr, self.steps_min)

      return x_best, acc, loss_best, x_best_adv

    def perturb(
      self,
      x_in: torch.Tensor,
      y_in: torch.Tensor,
      best_loss: bool = False,
      cheap: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
      x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
      y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

      adv = x.clone()
      acc = self.model(x).max(1)[1] == y
      if self.verbose:
        self.logger.info(
          f"-------------------------- running {self.norm}-attack"\
          f"with epsilon {self.eps:.4f} --------------------------"
        )
        self.logger.info(f"initial accuracy: {acc.float().mean:.2%}")
      startt = time.time()

      if not best_loss:
        if not cheap:
          raise ValueError("not implemented yet")

        else:
          for counter in range(self.n_restarts):
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0:
              ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
              x_to_fool, y_to_fool = (
                x[ind_to_fool].clone(),
                y[ind_to_fool].clone(),
              )
              (
                best_curr,
                acc_curr,
                loss_curr,
                adv_curr,
              ) = self.attack_single_run(
                x_to_fool, y_to_fool
              )  # nopep8
              ind_curr = (acc_curr == 0).nonzero().squeeze()
              #
              acc[ind_to_fool[ind_curr]] = 0
              adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
              if self.verbose:
                self.logger.info(
                  f"restart {counter} - "\
                  f"robust accuracy: {acc.float().mean():.2%} - cum. "\
                  f"time: {time.time() - startt:.1f} s"
                )

        return acc, adv

      else:
        adv_best = x.detach().clone()
        loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float("inf"))
        for counter in range(self.n_restarts):
          best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
          ind_curr = (loss_curr > loss_best).nonzero().squeeze()
          adv_best[ind_curr] = best_curr[ind_curr] + 0.0
          loss_best[ind_curr] = loss_curr[ind_curr] + 0.0

          if self.verbose:
            self.logger.info(
              f"restart {counter} - loss: {loss_best.sum().item():.5f}"
            )

        return loss_best, adv_best
