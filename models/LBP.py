
from timm.layers import Mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
import DecisionNCE
import math
import torch
import torch.nn as nn

from .components.MlpResNet import MlpResNet
from .components.ResNet import FilmResNet
from .components.ActionHead import BaseHead, DDPMHead
from .components.CrossAttn import CrossAttnBlock
from .MidPlanner import mid_planner_dnce_noise


class LBPPolicy(nn.Module):
    def __init__(
        self,
        imaginator_ckpt_path = None,
        proprio_input_dim = 9,
        proprio_hidden_dim = 32,
        vision_backbone_name: str = "resnet34",
        policy_num_blocks = 3,
        policy_hidden_dim = 256,
        action_size = 7,
        chunk_length = 4,
        num_views = 2,
        decoder_head = 'base',
        loss_func = nn.MSELoss,
        loss_func_conig = dict(reduction='mean'),
        recursive_step = 2,
        num_attn_layers = 3,
    ):
        super().__init__()
        # condition encoder
        self.imaginator = mid_planner_dnce_noise(recursive_step=4)
        self.imaginator.load_state_dict(torch.load(imaginator_ckpt_path, map_location='cpu'), strict=True)
        self.imaginator.requires_grad_(False)
        self.recursive_step = recursive_step
        self.latent_dim = 1024

        # context fusion
        self.goal_fusion = CrossAttnBlock(embed_dim=self.latent_dim, num_layers=num_attn_layers)

        # vision encoder
        self.num_views = num_views
        self.vision_encoder = FilmResNet(image_dim=3, cond_dim=self.latent_dim, backbone_name=vision_backbone_name)
        self.vision_dim = self.vision_encoder.vision_dim * self.num_views

        # proprio encoder
        self.proprio_dim = proprio_input_dim
        # self.proprio_encoder = Mlp(proprio_input_dim, proprio_hidden_dim, proprio_hidden_dim, norm_layer=nn.LayerNorm)

        # action decoder
        self.decoder_head = decoder_head
        if decoder_head == 'base':
            self.head = BaseHead(num_blocks=policy_num_blocks, input_dim=self.vision_dim + self.proprio_dim + self.latent_dim, 
                                hidden_dim=policy_hidden_dim, action_size=action_size * chunk_length)
        elif decoder_head == 'ddpm':
            self.head = DDPMHead(num_blocks=policy_num_blocks, input_dim=self.vision_dim + self.proprio_dim + self.latent_dim,  
                                hidden_dim=policy_hidden_dim, action_size=action_size * chunk_length)
        else :
            raise NotImplementedError

        # loss function
        self.loss_func = loss_func(**loss_func_conig)

    def forward_cond(self, cur_images, instruction):
        planned_subgoals, details = self.imaginator.generate(cur_images, instruction, self.recursive_step)
        lang_embeding = details['lang_latent']
        cur_query = details['img_latent']
        fused_goal = self.goal_fusion(cur_query.unsqueeze(1), planned_subgoals).squeeze(1)
        return lang_embeding, fused_goal

    def forward_obs(self, cur_images, cur_proprios, conditions, fused_goal):
        B, V, C, H, W = cur_images.shape
        cond = conditions.unsqueeze(1).repeat(1, V, 1).reshape(B*V, -1)
        vision_obs = cur_images.reshape(B*V, C, H, W) # B*2 3 224 224
        vision_semantics = self.vision_encoder(vision_obs, cond)
        vision_semantics = vision_semantics.reshape(B, -1)
        all_obs = torch.cat([vision_semantics, cur_proprios, fused_goal], dim=-1)
        return all_obs
    
    def forward_head(self, all_obs, cur_actions):
        if self.decoder_head == 'base':
            pred_actions = self.head(all_obs)
            loss = self.loss_func(pred_actions, cur_actions)
            return loss
        elif self.decoder_head == 'ddpm':
            noise, t_tensor, noise_action, noise_pred = self.head(all_obs, cur_actions)
            loss = (((noise_pred - noise) ** 2).sum(axis = -1)).mean()
            return loss

    def generate_head(self, all_obs):
        if self.decoder_head == 'base':
            return self.head.generate(all_obs)
        elif self.decoder_head == 'ddpm':
            return self.head.generate(all_obs)

    def forward(self, cur_images, cur_proprios, cur_actions, instruction, **kwargs):
        conditions, fused_goal = self.forward_cond(cur_images, instruction)
        all_obs = self.forward_obs(cur_images, cur_proprios, conditions, fused_goal)
        loss = self.forward_head(all_obs, cur_actions)
        # Note: 'loss' must be in the dict
        return loss, dict(loss=loss)
    
    def generate(self, cur_images, cur_proprios, instruction, **kwargs):
        conditions, fused_goal = self.forward_cond(cur_images, instruction)
        all_obs = self.forward_obs(cur_images, cur_proprios, conditions, fused_goal)
        pred_actions = self.generate_head(all_obs)
        # Note: 'actions' must be in the dict
        return pred_actions, dict(actions=pred_actions)
    


def lbp_policy_ddpm_res18_libero(imaginator_ckpt_path, chunk_length=6, recursive_step=2, **kwargs):
    return LBPPolicy(proprio_input_dim=9, proprio_hidden_dim=32, vision_backbone_name="resnet18", decoder_head='ddpm',
                    num_attn_layers=3, recursive_step=recursive_step, imaginator_ckpt_path=imaginator_ckpt_path,
                    policy_num_blocks=3, policy_hidden_dim=256, action_size=7, chunk_length=chunk_length)


def lbp_policy_ddpm_res34_libero(imaginator_ckpt_path, chunk_length=6, recursive_step=2, **kwargs):
    return LBPPolicy(proprio_input_dim=9, proprio_hidden_dim=32, vision_backbone_name="resnet34", decoder_head='ddpm',
                    num_attn_layers=3, recursive_step=recursive_step, imaginator_ckpt_path=imaginator_ckpt_path,
                    policy_num_blocks=3, policy_hidden_dim=256, action_size=7, chunk_length=chunk_length)
