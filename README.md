# DRGen
Official Repository for the MICCAI 2022 paper titled DRGen: Domain Generalization in Diabetic Retinopathy Classification

# Abstract:
Domain  Generalization  is  a  challenging  problem  in  deep learning especially in medical image analysis because of the huge diversity between different datasets. Existing papers in the literature tend to optimize performance on single target domains, without regards to model generalizability on other domains or distributions. High discrepancy in the number of images and major domain shifts, can therefore cause single-source trained models to under-perform during testing. In this paper, we address the problem of domain generalization in Diabetic Retinopathy (DR) classification. The baseline for comparison is set as joint training on different datasets, followed by testing on each dataset individually. We therefore introduce a method that encourages seeking a flatter minima during training while imposing a regularization. This reduces gradient variance from different domains and therefore yields satisfactory results on out-of-domain DR classification. We show that adopting DR-appropriate augmentations enhances model performance and in-domain generalizability. By performing our evaluation on 4 open-source DR datasets, we show that the proposed domain generalization method outperforms separate and joint training strategies as well as well-established methods.

The application code we use is based on backbone codes from both SWAD(domainbed)[1] and Fishr [2].


# References
1. Cha, J., Cho, H., Lee, K., Park, S., Lee, Y., Park, S.: Domain generalization needs stochastic weight averaging for robustness on domain shifts. CoRR arXiv:2102.08604 (2021)
2. Ramé, A., Dancette, C., Cord, M.: Fishr: invariant gradient variances for out-of-distribution generalization. CoRR arXiv:2109.02934 (2021)

