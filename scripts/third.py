import torch
import torch.nn as nn
import torch.optim as optim
from general_utils import AttrDict, make_image_seq_strip
from sprites_datagen.moving_sprites import DistractorTemplateMovingSpritesGenerator, AgentTemplateMovingSpritesGenerator
from sprites_datagen.rewards import ZeroReward, VertPosReward, HorPosReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward
import numpy as np
torch.autograd.set_detect_anomaly(True)
import os
import wandb
import cv2
import matplotlib.pyplot as plt

    
save_dir = "../model_weights"
os.makedirs(save_dir, exist_ok=True)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data, 0, 0.001)

def findL2(x):
    return np.sqrt((np.array(x["agent_x"])-np.array(x["target_x"]))**2 
                    +(np.array(x["agent_y"])-np.array(x["target_y"]))**2)


class Encoder(nn.Module):
    def __init__(self,neck=5):
        super(Encoder, self).__init__()
        self.neck=neck
        self.pipe = nn.Sequential(nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(64, self.neck))

    def forward(self, x):
        x = self.pipe(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fcs = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.fcs(x)
        return x

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out

class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape 

    def forward(self, x):
        return x.view(self.shape)

class Decoder(nn.Module):
    def __init__(self,neck):
        super(Decoder, self).__init__()
        self.neck=neck
        self.pipe = nn.Sequential(
            nn.Linear(self.neck,64),
            Reshape(64,1,1),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(self.neck)

        x = self.pipe(x)
        return x

class RewardHead(nn.Module):
    def __init__(self, input_dim):
        super(RewardHead, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.fcs(x)
        return x

class FullModel(nn.Module):
    def __init__(self, conditioning_frames=3,neck=5):
        super(FullModel, self).__init__()
        self.neck=neck
        self.encoder = Encoder(self.neck)
        self.mlp = MLP(input_dim=3*self.neck, hidden_dim=32, output_dim=32)
        self.predictor = Predictor(input_dim=32, hidden_dim=32)
        self.reward_head = RewardHead(input_dim=32)
        self.lstm_input_list = []
        self.conditioning_frames = conditioning_frames

    def forward(self, x):
        z = self.encoder(x)
        mlp_out = self.mlp(z)
        mlp_output = mlp_out.clone()
        self.lstm_input_list.append(mlp_output)

        if len(self.lstm_input_list) > self.conditioning_frames:
            self.lstm_input_list.pop(0)
            lstm_in = torch.stack(self.lstm_input_list).unsqueeze(0)
            #assert lstm_in.requires_grad
            lstm_out = self.predictor(lstm_in)
            #assert lstm_out.requires_grad
            rewards = self.reward_head(lstm_out)
            #assert rewards.requires_grad
            rewards = rewards.flatten(start_dim=0)
            #assert rewards.requires_grad
            return rewards, z

        return torch.zeros((self.conditioning_frames), requires_grad = True), z



class WeightedMSELoss(nn.Module):
    def __init__(self, weight_for_white=10.0):
        super(WeightedMSELoss, self).__init__()
        self.weight_for_white = weight_for_white  # Weight for white pixels

    def forward(self, input, target):
        
        weights = torch.ones_like(target) + target*(self.weight_for_white-1)
        # Compute the element-wise squared error
        squared_error = (input - target) ** 2

        # Apply weights
        weighted_squared_error = squared_error * weights

        # Return the mean of the weighted squared error
        return weighted_squared_error.mean()


class MainModule():
    def __init__(self, checkpoint_path = None, ww=2, neck=5, num_epoch=20):
        """self.spec = AttrDict(
        resolution=64,
        max_seq_len=500,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=2,
        rewards=[VertPosReward, HorPosReward, AgentXReward, AgentYReward, TargetXReward, TargetYReward]
    )
        self.gen = DistractorTemplateMovingSpritesGenerator(self.spec)"""
        self.num_epoch = num_epoch
        self.spec = AttrDict(
        resolution=64,
        max_seq_len=500,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=1,
        rewards=[VertPosReward, HorPosReward]
    )
        self.gen = AgentTemplateMovingSpritesGenerator(self.spec)
        self.prediction_horizon = 1
        self.num_enc = 3
        self.device = "cpu"
        self.neck=neck
        self.model = FullModel(conditioning_frames=self.prediction_horizon,neck=self.neck).to(self.device)
        self.model.apply(init_weights)
        self.decoder = Decoder(self.neck).to(self.device)
        self.decoder.apply(init_weights)
        self.optim = optim.Adam(self.model.parameters(),  lr=0.0001, betas=(0.9, 0.999))
        self.recon_optim = optim.Adam(self.decoder.parameters(),  lr=0.0002, betas=(0.9, 0.999))
        if checkpoint_path == None:
            print("No checkpoint")
        else:
            checkpoint = torch.load(os.path.join(save_dir, checkpoint_path))  
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.weight_for_white=ww
        self.recon_loss_fn = WeightedMSELoss(weight_for_white=self.weight_for_white)
        self.loss_fn = nn.MSELoss()
        self.reward_loss_list=[]
        self.recon_loss_list=[]
        wandb.init(
        project="representation_RL",
    
        config={
        "epochs": self.num_epoch,
        "white_weight": self.weight_for_white,
        "bottlenck": self.neck
        })

        

    def do(self, recon_epoch=10):
        
        for epoch in range(self.num_epoch):
            traj = self.gen.gen_trajectory()
            torch_images = torch.stack([torch.tensor(img, dtype=torch.float32).unsqueeze(0) for img in traj.images]).to(self.device) / 255
            #rewards = findL2(traj.rewards)
            rewards = traj.rewards["vertical_position"]
            acc_reward_loss = 0
            acc_recon_loss = 0

            for t in range(0, len(torch_images)-self.num_enc-self.prediction_horizon):
                self.optim.zero_grad()
                self.recon_optim.zero_grad()
                observation = torch_images[t:t+self.num_enc]
                pred_rewards, z = self.model(observation)
                true_rewards = rewards[t+self.num_enc:t+self.num_enc+self.prediction_horizon]
                true_rewards_tensor = torch.tensor(true_rewards, dtype=torch.float32).to(self.device)
                
                reward_loss = self.loss_fn(pred_rewards, true_rewards_tensor).to(self.device)
                assert pred_rewards.requires_grad
                

                z_dec=z[0].clone().detach()
                assert z_dec.requires_grad == False
                decoded_image = self.decoder(z_dec)
                recon_loss = self.recon_loss_fn(decoded_image, torch_images[t])
                wandb.log({"recon_loss": recon_loss})
                wandb.log({"reward_loss": reward_loss})
                acc_reward_loss += reward_loss.clone().detach()
                acc_recon_loss += recon_loss.clone().detach()

                if epoch < recon_epoch:
                    reward_loss.backward()
                    self.optim.step()
                else:
                    recon_loss.backward()
                    self.recon_optim.step()
                
                if t % 100 == 99:
                    self.reward_loss_list.append(acc_reward_loss)
                    acc_reward_loss=0
                    self.recon_loss_list.append(acc_recon_loss)
                    acc_recon_loss=0
                    print(f"Epoch {epoch}, timestep {t}, reward_loss: {reward_loss}, recon_loss: {recon_loss}")
                    image = decoded_image.squeeze().detach().cpu().numpy() 
                    original_image = torch_images[t+3].squeeze().detach().cpu().numpy()
                    assert image.shape == original_image.shape
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    image = cv2.resize(image, (512, 512)) # *255
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
                    original_image = cv2.resize(original_image, (512, 512)) # *255
                    cv2.imshow("Original Image", original_image)
                    cv2.imshow("Reconstructed Image", image)

                    og_image = wandb.Image(original_image, caption=f"og_epoch {epoch} timestep {t}")
                    rec_image = wandb.Image(image, caption=f"og_epoch {epoch} timestep {t}")
                    wandb.log({"original_image":og_image})
                    wandb.log({"recon_image":rec_image})

                    cv2.waitKey(1)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break

        plt.plot(self.reward_loss_list, label='Reward Loss')
        plt.plot(self.recon_loss_list, label='Recon Loss')
        plt.legend()
        plt.show()
    
if __name__ == "__main__":
    m=MainModule(num_epoch=50, ww=15, neck = 5)
    m.do(recon_epoch=25)


