from warpgan import WarpGAN

model_dir = "./pretrained/warpgan_pretrained"
network = WarpGAN()
network.load_model(model_dir)