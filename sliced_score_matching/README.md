Install requirements are the same as https://github.com/ermongroup/sliced_score_matching
I ignored the version numbers since they are quite old and it works fine. 

To train:
  python main.py --runner VAERunner --config vae/cifar10_vae_8.yml
  
To calculate FID:
  python main.py --runner VAERunner --config vae/cifar10_vae_8.yml --test_fid
