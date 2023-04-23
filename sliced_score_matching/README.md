Install requirements are the same as https://github.com/ermongroup/sliced_score_matching
I ignored the version numbers since they are quite old and it works fine. (scipy==1.2.1 is needed i think)

To train:
  python main.py --runner VAERunner --config vae/celeba_vae.yml
  
To calculate FID (ID = 10, 40. 70 or 100):

  python main.py --runner VAERunner --config vae/celeba.yml --test_fid --checkpoint_id ID
  
 Replace vae/celeba_vae.yml with the different configs in the configs/ folder.
  
 # CELEBA dataset download
  
  mkdir  Score-based-Matching-Density-Estimation/sliced_score_matching/datasets/
  
 Download archive.zip from:
  https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?resource=download
  and place it in  "Score-based-Matching-Density-Estimation/sliced_score_matching/datasets/"
  
  cd  Score-based-Matching-Density-Estimation/sliced_score_matching/datasets/
  
  unzip archive.zip
  
  mv img_align_celeba celeba

  mv celeba/img_align_celeba celeba/celeba
  
  mv list_* celeba/
