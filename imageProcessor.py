from tqdm import tqdm
import numpy as np
import dim_reduce as dm

class Image_Processor():

  def __init__(self, images):

    self._update_images(images)

  def _update_images(self, images):

    self.images = images
    self.img_num = len(self.images)
    self.img_dim = self.images[0].shape

  def check_rbg(self):

    # determine if all r,g,b intensities are the same for each pixel
    rbgdim = self.img_dim[2]
    equal_pix = 0; non_equal_pix = 0
    for i, j in enumerate(self.images):
      for row in range(rbgdim):
        for col in range(rbgdim):
          val = j[row,col,0]
          rbg = j[row,col,:]
          if (all(pix == val for pix in rbg)):
            equal_pix += 1
          else:
            non_equal_pix += 1
    return (equal_pix, non_equal_pix)

  def greyscale(self):
    # turn r,b,g images to greyscale if r,b,g intensities are all the same 
    self._update_images([img[:,:,0] for img in self.images])

  def normalize(self):
    self._update_images([img*(.255/img.max()) for img in self.images])

  def stack(self):
    # concatenate images for shape of (data_len,img_dim_1,img_dim_2,color_dim)
    stack = np.empty((self.img_num, *self.img_dim))
    with tqdm(total = self.img_num) as pbar: 
      for i, img in enumerate(self.images):
        stack[i] = img
        pbar.update(1) 
    self._update_images(stack)

  def SVD(self, varExpl = 0.95, train_ratio = 1):
    # perform SVD to reduce dimensionality of images 
    train_index = np.random.choice(self.img_num, 
      int(np.ceil(self.img_num * train_ratio)), 
      replace=False)
    train_images = [self.images[i] for i in train_index]

    print(' ----- Training SVD ----- ')
    svd_images, var_expl, num_pcs = dm.train_sparse_SVD(
      images = train_images,
      n_components = self.img_dim[1]-1,
      var_explain = varExpl)
    svd_dim = int(np.ceil(np.mean(num_pcs)))

    print(' ----- Testing Left SVD ----- ')
    svd_images, var_expl = dm.get_sparse_SVD(images = self.images,
      n_components = svd_dim)
    unaltered_img_dim = np.copy(self.img_dim)
    self._update_images(svd_images)

    perc_reduced = (1-(np.multiply(*self.img_dim)/(np.multiply(*unaltered_img_dim))))*100
    print('SVD Dimension: {} --- {} % compression'.format(
      self.img_dim, round(perc_reduced, 2)))

  def getSVD(self, n_components):
    svd_images, var_expl = dm.get_sparse_SVD(images = self.images,
      n_components = n_components)
    self._update_images(svd_images)

  def transpose(self):
    self._update_images([np.transpose(img, (1,0)) for img in self.images])