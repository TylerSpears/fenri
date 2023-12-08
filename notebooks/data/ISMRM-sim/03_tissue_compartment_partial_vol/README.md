# Notes on Partial Volume Fraction Estimation

ISMRM 2015 only included component volume fraction maps in 2.00mm resolution, and it is
not described how those fraction maps were generated. However, the given T1w-like image
is in 1.00mm resolution, which is better, and we can theoretically have a one-to-one and
onto mapping from T1 values to tissue volume fractions, given that these are simulations.

## 1. Tissue Mask Selection/Generation

A brain mask/tissue mask needs to be generated. One can use the given tissue mask, or segment the given T1, or use the tissue mask from HCP subject 100307.

## 2. Downsampling for Learning

Downsample the ISMRM T1w and the chosen tissue mask to 2.00mm isotropic. This allows matching the voxel values between the T1 and provided volume fraction maps.

## 3. Regression

Train a simple regression model to give a smooth mapping between the downsampled T1w image values and the provided CSF volume fraction.

One example solution would use the following:

```python
# sklearn version 1.1.3

lr_t1_v = lr_t1[lr_tissue_mask]
lr_csf_v = lr_pv_csf[lr_tissue_mask]

lr_pv_sum_mask = (lr_csf_v + lr_gm_v + lr_wm_v) > 0.99
lr_t1_v = lr_t1_v[lr_pv_sum_mask]
lr_csf_v = lr_csf_v[lr_pv_sum_mask]

lr_t1_sort = np.argsort(lr_t1_v)f

# Subset data for faster training time.
x = lr_t1_v[lr_t1_sort].reshape(-1, 1)[::30]
y = lr_csf_v[lr_t1_sort][::30]

# Important: the default epsilon of 0.1 is too low, this allows for too much deviation
# from the training samples.
model = sklearn.svm.SVR(kernel='rbf', max_iter=100000, epsilon=0.01)
model.fit(x, y)
# for checkin the model:
# pred = model.predict(lr_t1_v[lr_t1_sort].reshape(-1, 1))
# ...
```

We have also included an IPython notebook `regress.ipynb` that contains a complete regression example with visualizations for model performance. Additionally, we have included the trained model parameters for the SVM model used in the creation of this dataset in the `.pickle` file.

## 4. Denoising T1

Noise was injected into the provided T1 image, so remove as much as possible would be ideal for a clean simulation. One possible denoiser that seems to work well is the ANTS non-local means denoiser. For example:

```bash
"${ANTSPATH}/DenoiseImage" -d 3 \
    -i ismrm-2015_t1w.nii.gz \
    -n Rician \
    -s 1 \
    -v \
    -o denoised_t1w.nii.gz
```

## 5. Prediction

Run inference with the trained model on the *denoised* T1 image to get the final prediction.
