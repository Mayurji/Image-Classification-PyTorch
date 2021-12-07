"""
Acknowledgement: https://github.com/davda54/sam

From paper:

- Sharpness-Aware Minimization for Efficiently Improving Generalization
- ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks

Idea behind SAM (Sharpness Aware Minimization):

- Its motivation comes from the relation between the landscape of the loss function and generalization.
- A regular model finds a parameter W with low training loss value but with SAM, it finds parameter
whose neighborhood of parameters have uniformly low training loss value.
- It improves model generalization by simultaneously minimizing loss value and loss sharpness.
- It seeks parameters that lie in neighborhood having uniformly low loss value.
- 

"""
import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho < 0:
            raise Exception("rho should not be negative")

        defaults = dict(rho=rho, adaptive = adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        """
        Equation 2's denominator from the paper, calculating the norm.
        """
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group['params']
                if p.grad is not None
                ]),p=2)

        return norm

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            # rho - neighborhood size
            scale = group['rho'] / (grad_norm + 1e-12)

            for p in group['params']:
                if p.grad is None: continue
                self.state[p]['old_p'] = p.data.clone()
                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale.to(p)
                p.add(e_w) # Climb to local maximum 'w + e(w)'

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"] #get back w from "w + e(w)"

        self.base_optimizer.step() #Sharpness Aware Update

        if zero_grad: self.zero_grad()