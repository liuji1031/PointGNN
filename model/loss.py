import torch
from torch.nn import NLLLoss, HuberLoss

class Loss(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """
    def __init__(self, **kwargs):
        """_summary_

        Args:
            kwargs: _description_
        """
        super().__init__()
        self.background_weights = torch.tensor(kwargs.get('background_weights', [1., 1.]))
        self.background_loss = NLLLoss(weight=self.background_weights, reduction='mean')

        self.object_weights = kwargs.get('object_weights', None)
        if self.object_weights is not None:
            self.object_weights = torch.tensor(self.object_weights)
        self.object_loss = NLLLoss(weight=self.object_weights,reduction='none')

        self.huber_loss = HuberLoss(reduction='none')
        self.loss_coeff = kwargs.get('loss_coeff', {'background': 1., 'object': 1., 'xyz': 1., 'lwh': 1., 'r': 1.})

    def forward(self, background_pred,
                object_pred, object_target,
                xyz_pred, xyz_target,
                lwh_pred, lwh_target,
                r_pred, r_target,
                positive_mask):
        """_summary_

        Args:
            pred (_type_): _description_
            target (_type_): _description_

        Returns:
            _type_: _description_
        """
        background_target = positive_mask.to(torch.int16)
        background_loss = self.background_loss(background_pred, background_target)

        object_loss = self.object_loss(object_pred, object_target)
        object_loss = (object_loss*positive_mask).mean()

        xyz_loss = self.huber_loss(xyz_pred, xyz_target)
        xyz_loss = (xyz_loss*positive_mask[:,torch.newaxis]).mean()

        lwh_loss = self.huber_loss(lwh_pred, lwh_target)
        lwh_loss = (lwh_loss*positive_mask[:,torch.newaxis]).mean()

        r_loss = self.huber_loss(r_pred, r_target)
        r_loss = (r_loss*positive_mask[:,torch.newaxis]).mean()

        total_loss = self.loss_coeff['background']*background_loss + \
                     self.loss_coeff['object']*object_loss + \
                     self.loss_coeff['xyz']*xyz_loss + \
                     self.loss_coeff['lwh']*lwh_loss + \
                     self.loss_coeff['r']*r_loss

        return {"total_loss": total_loss,
                "background_loss": background_loss,
                "object_loss": object_loss,
                "xyz_loss": xyz_loss,
                "lwh_loss": lwh_loss,
                "r_loss": r_loss}


