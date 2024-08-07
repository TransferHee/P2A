import torch
import torch.nn.functional as F

from captum.attr import GuidedGradCam

from .utils import load_single_model, wrap_model

class P2A():
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1, num_ens=30, tau=1e+1, eta=0.3,
                device='cpu', attack='P2A', feature_layer='layer2'):
        
        self.attack = attack
        self.epsilon = epsilon
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.device = device
        
        self.model = self.load_model(model_name)
        
        self.feature_layer = self.find_layer(feature_layer)
        
        self.explainer = GuidedGradCam(self.model, layer=self.get_last_feature_layer(self.model, model_name))
        
        self.num_ens = num_ens
        self.tau = tau
        self.eta = eta
        
    def load_model(self, model_name):
        model = load_single_model(model_name)
        return wrap_model(model.eval.to(self.device))
    
    def find_layer(self,layer_name):
        parser = layer_name.split(' ')
        m = self.model[1]
        for layer in parser:
            if layer not in m._modules.keys():
                print("Selected layer is not in Model")
                exit()
            else:
                m = m._modules.get(layer)
        return m
  
    def get_last_feature_layer(self, model, model_name):
        if model_name == 'resnet101':
            return model[1].layer4
        else:
            raise NotImplementedError('Model not supported')
    
    def __forward_hook(self, model, input, output):
        global mid_output
        mid_output = output

    def __backward_hook(self, model, input, output):
        global mid_grad
        mid_grad = output

    def sample_gumbel(self, logits, eps=1e-20):
        U = torch.rand_like(logits)
        return -torch.log(-torch.log(U + eps) + eps)
        
    def random_prioritized_patch_mask(self, data, priority):
        """
        data: (N, C, H, W) tensor for input images
        priority: (N, C, P_h, P_w) pooled tensor from XAI
        
        return: (N, C, H, W) tensor for Prioiritized Patch Mask
        """
    
        p_prime = priority + self.sample_gumbel(priority)
        p_prime = torch.sigmoid(p_prime / self.tau)
        
        s = 1 - self.eta
        p_hat = torch.max(s - p_prime, torch.zeros_like(p_prime))
        
        mask = F.interpolate(p_hat, size=(data.shape[2], data.shape[3]), mode='nearest')
        
        return mask
    
    def get_agg_grad(self, data, label):
        x = torch.zeros(data.size()).cuda()
        x.copy_(data).detach()
        x.requires_grad = True
        
        exp = self.explainer.attribute(data, target=label).detach()
        assert data.size() == exp.size()
        
        h2 = self.feature_layer.register_full_backward_hook(self.__backward_hook)
        agg_grad = 0
        for l in range(self.num_ens):
            if l % 4 == 0:
                priority = exp
            elif l % 4 == 1:
                priority = F.max_pool2d(exp, kernel_size=3, stride=3)
            elif l % 4 == 2:
                priority = F.max_pool2d(exp, kernel_size=5, stride=5)
            else:
                priority = F.max_pool2d(exp, kernel_size=7, stride=7)
            
            mask = self.random_prioritized_patch_mask(data, priority)
            
            output_random = self.model(x*mask)
            output_random = torch.softmax(output_random, 1)
            
            loss = 0
            for batch_i in range(data.shape[0]):
                loss += output_random[batch_i][label[batch_i]]

            self.model.zero_grad()
            loss.backward()
            agg_grad += mid_grad[0].detach()

        for batch_i in range(data.shape[0]):
            agg_grad[batch_i] /= agg_grad[batch_i].norm(2)
            
        h2.remove()
        return agg_grad
    
    def forward(self, data, label):
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = torch.zeros_like(data).to(self.device)
        delta.requires_grad = True

        h = self.feature_layer.register_forward_hook(self.__forward_hook)

        agg_grad = self.get_agg_grad(data, label)

        momentum = 0
        for _ in range(self.epoch):
            logits = self.model(data + delta)
            loss = (agg_grad * mid_output).sum()
            self.model.zero_grad()
            
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            momentum = momentum * self.decay + grad / (grad.abs().mean(dim=(1,2,3), keepdim=True))

            # l-inf norm
            delta = torch.clamp(delta - self.alpha * torch.sign(momentum), -self.epsilon, self.epsilon)
            delta = torch.clamp(delta, 1.0 - data, 0.0 - data)

        h.remove()
        return delta.detach()