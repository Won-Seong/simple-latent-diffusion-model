# Welcome to the Simple Latent Diffusion Model

🌐 README in Korean: [KR 한국어 버전](README_ko.md)

This repository contains a simplified implementation of a latent diffusion model. The code and contents will be updated continuously.
| Dataset                                     | Generation Process of Latents           | Generated Data                          |
|---------------------------------------------|-----------------------------------------|-----------------------------------------|
| Swiss-roll  | <img src="assets/swiss_roll.gif" width="300"/>   | <img src="assets/swiss_roll_image.png" width="300"/>  |
| CIFAR-10  | <img src="assets/cifar10.gif" width="300"/>   | <img src="assets/cifar10_image.png" width="300"/>  |
| CelebA  | <img src="assets/celeba.gif" width="300"/>   | <img src="assets/celeba_image.png" width="300"/>  |

## Generate Composite with CLIP

The following table shows text-to-image generation with CLIP. The dataset is [Asian Composite Dataset](https://aihub.or.kr/).

| Text(Eng) | Text(Kor)      | Generation Process of Latents           | Generated Data                          |
|---------------------------------------------|----|-----------------------------------------|-----------------------------------------|
| A round face with voluminous, slightly long short hair, along with barely visible vocal cords, gives off a more feminine aura than a masculine one. The well-defined eyes and lips enhance the subject's delicate features, making them appear more refined and intellectual. | 동그란 얼굴에 풍성하고 살짝 긴 커트머리와 거의 나오지 않은 성대가 남성보다는 여성쪽의 분위기를 냅니다. 또렷한 눈과 입술이  인물의 섬세함을 더 증진시키고 지적이게 보이게 도와줍니다. | <img src="assets/swiss_roll.gif" width="300"/>   | <img src="assets/swiss_roll_image.png" width="300"/>  |
| The hairstyle seems somewhat unpolished, lacking a sophisticated touch. The slightly upturned eyes give off a sharp and somewhat sensitive impression. Overall, they appear to have a slender physique and seem capable of handling tasks efficiently, but their social interactions might not be particularly smooth. | 헤어 손질은 좀 미숙하게 하여 세련되어 보이지는 않는다. 눈끝이 올라 가서 그런지 눈빛이 좀 날카롭고 예민해 보인다. 전체적으로 체격도 마른형일 것 같고 일처리는 잘할 것 같지만 교우관계는 그리 편할 것 같지는 않다. | <img src="assets/cifar10.gif" width="300"/>   | <img src="assets/cifar10_image.png" width="300"/>  |






## Tutorials

- [Tutorial for Latent Diffusion Model](notebook/simple_latent_diffusion_model_tutorial.ipynb)

## Usage

The following example demonstrates how to use the code in this repository.

```python
import torch
import os

from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
from helper.data_generator import DataGenerator
from helper.painter import Painter
from helper.trainer import Trainer
from helper.loader import Loader
from diffusion_model.models.latent_diffusion_model import LatentDiffusionModel
from diffusion_model.network.uncond_u_net import UnconditionalUnetwork
from diffusion_model.sampler.ddim import DDIM

# Path to the configuration file
CONFIG_PATH = './configs/cifar10_config.yaml'

# Instantiate helper classes
painter = Painter()
loader = Loader()
data_generator = DataGenerator()

# Load CIFAR-10 dataset
data_loader = data_generator.cifar10(batch_size=128)

# Train the Variational Autoencoder (VAE)
vae = VariationalAutoEncoder(CONFIG_PATH)  # Initialize the VAE model
trainer = Trainer(vae, vae.loss)  # Create a trainer for the VAE
trainer.train(dl=data_loader, epochs=1000, file_name='vae', no_label=True)  # Train the VAE

# Train the Latent Diffusion Model (LDM)
sampler = DDIM(CONFIG_PATH)  # Initialize the DDIM sampler
network = UnconditionalUnetwork(CONFIG_PATH)  # Initialize the U-Net network
ldm = LatentDiffusionModel(network, sampler, vae, image_shape=(3, 32, 32))  # Initialize the LDM
trainer = Trainer(ldm, ldm.loss)  # Create a trainer for the LDM
trainer.train(dl=data_loader, epochs=1000, file_name='ldm', no_label=True)  
# Train the LDM; set 'no_label=False' if the dataset includes labels

# Generate samples using the trained diffusion model
ldm = LatentDiffusionModel(network, sampler, vae, image_shape=(3, 32, 32))  # Re-initialize the LDM
loader.model_load('./diffusion_model/check_points/ldm_epoch1000', ldm, ema=True)  # Load the trained model
sample = ldm(n_samples=4)  # Generate 4 sample images
painter.show_images(sample)  # Display the generated images

```


## References
- [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
- [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)
- [labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/stable_diffusion)
