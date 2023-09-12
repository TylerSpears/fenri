function [] = spm_lst_lesion_seg(spm_dir, t1_f, flair_f, kappa, phi, maxiter)
    addpath(fullfile(spm_dir, 'toolbox', 'LST'))
    ps_LST_lga(t1_f, flair_f, kappa, maxiter, phi, 0)
end
