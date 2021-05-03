import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable

from collections import OrderedDict
from maml.utils import gradient_update_parameters
from maml.utils import tensors_to_device, compute_accuracy
from maml.min_norm_solvers import MinNormSolver
from .maml import MAML

__all__ = ['Pareto_MAML']


class Pareto_MAML(MAML):
    """Meta-learner class for Model-Agnostic Meta-Learning [1].

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    optimizer : `torch.optim.Optimizer` instance, optional
        The optimizer for the outer-loop optimization procedure. This argument
        is optional for evaluation.

    step_size : float (default: 0.1)
        The step size of the gradient descent update for fast adaptation
        (inner-loop update).

    first_order : bool (default: False)
        If `True`, then the first-order approximation of MAML is used.

    learn_step_size : bool (default: False)
        If `True`, then the step size is a learnable (meta-trained) additional
        argument [2].

    per_param_step_size : bool (default: False)
        If `True`, then the step size parameter is different for each parameter
        of the model. Has no impact unless `learn_step_size=True`.

    num_adaptation_steps : int (default: 1)
        The number of gradient descent updates on the loss function (over the
        training dataset) to be used for the fast adaptation on a new task.

    scheduler : object in `torch.optim.lr_scheduler`, optional
        Scheduler for the outer-loop optimization [3].

    loss_function : callable (default: `torch.nn.functional.cross_entropy`)
        The loss function for both the inner and outer-loop optimization.
        Usually `torch.nn.functional.cross_entropy` for a classification
        problem, of `torch.nn.functional.mse_loss` for a regression problem.

    device : `torch.device` instance, optional
        The device on which the model is defined.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Li Z., Zhou F., Chen F., Li H. (2017). Meta-SGD: Learning to Learn
           Quickly for Few-Shot Learning. (https://arxiv.org/abs/1707.09835)

    .. [3] Antoniou A., Edwards H., Storkey A. (2018). How to train your MAML.
           International Conference on Learning Representations (ICLR).
           (https://arxiv.org/abs/1810.09502)
    """
    def __init__(self, model, optimizer=None, step_size=0.1, first_order=False,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=F.cross_entropy, device=None):
        
        super(Pareto_MAML, self).__init__(model, optimizer, step_size, first_order,
                 learn_step_size, per_param_step_size,
                 num_adaptation_steps, scheduler,
                 loss_function, device)

    
    def get_outer_loss(self, batch, train=True):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        is_classification_task = (not test_targets.dtype.is_floating_point)
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.num_adaptation_steps,
                num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.
        }
        if is_classification_task:
            results.update({
                'accuracies_before': np.zeros((num_tasks,), dtype=np.float32),
                'accuracies_after': np.zeros((num_tasks,), dtype=np.float32)
            })
        
        
        grads = {}
        losses = {}
        mean_outer_loss = torch.tensor(0., device=self.device)
        
        for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
                in enumerate(zip(*batch['train'], *batch['test'])):
            params, adaptation_results = self.adapt(train_inputs, train_targets,
                is_classification_task=is_classification_task,
                num_adaptation_steps=self.num_adaptation_steps,
                step_size=self.step_size, first_order=self.first_order)

            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']
            if is_classification_task:
                results['accuracies_before'][task_id] = adaptation_results['accuracy_before']

            with torch.set_grad_enabled(self.model.training):
                test_logits = self.model(test_inputs, params=params)
                outer_loss = self.loss_function(test_logits, test_targets)
                results['outer_losses'][task_id] = outer_loss.item()
                losses[task_id] = outer_loss
            
            if train:
                outer_loss.backward(retain_graph=True)
                grads[task_id] = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        grads[task_id].append(Variable(param.grad.data.clone(), requires_grad=False))

                # print(grads[task_id])
                self.model.zero_grad()
                # print(grads[task_id])
                
                # for param in self.model.parameters():
                #     print(param.grad.data.clone())

            if is_classification_task:
                results['accuracies_after'][task_id] = compute_accuracy(
                    test_logits, test_targets)

        if train:
            grad_list = [grads[t] for t in grads.keys()]
            sol, min_norm = MinNormSolver.find_min_norm_element(grad_list)

            for i, key in enumerate(grads.keys()):
                mean_outer_loss += float(sol[i])*losses[key]

        
        else:
            for key in losses.keys():
                mean_outer_loss += losses[key]
            mean_outer_loss.div_(num_tasks)

        results['mean_outer_loss'] = np.mean([losses[key].item() for key in losses.keys()])
        
        return mean_outer_loss, results



