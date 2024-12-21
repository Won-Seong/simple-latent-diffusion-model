import torch
from auto_encoder.models.variational_auto_encoder import VariationalAutoEncoder
import os

from auto_encoder.helper.data_generator import DataGenerator
from auto_encoder.helper.painter import Painter
from auto_encoder.helper.trainer import Trainer
from auto_encoder.helper.loader import Loader

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device : {device}\t'  + (f'{torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU' ))
    dg = DataGenerator()
    dl = dg.cifar10(batch_size = 128)
    pt = Painter()
    
    a = next(iter(dl))[0][0:2]
    pt.show_or_save_images(a)
    
    ae = VariationalAutoEncoder(config_path = './auto_encoder/configs/cifar10_config.yaml',embed_dim=3).to(device)
    
    d, post = ae(a.to(device))
    pt.show_or_save_images(d)

    ld = Loader()
    ld.load_with_acc(file_name = './auto_encoder/check_points/embed3_epoch51',model = ae)

    
    #tr = Trainer(ae, loss_fn = ae.loss, no_label=True)
    #tr.accelerated_train(dl, 300, './auto_encoder/check_points/embed3')
    d, post = ae(a.to(device))
    print(post.sample())
    pt.show_or_save_images(d)
    
    d = ae(a.to(device) + torch.randn_like(a).to(device) * 0.1)[0]
    pt.show_or_save_images(d)
    
    d = ae.decode( torch.randn(10, 3, 8, 8).to(device) )
    pt.show_or_save_images(d)
    
    d = ae.decode( torch.zeros(10, 3, 8, 8).to(device) )
    pt.show_or_save_images(d)
   
   