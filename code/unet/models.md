

| job | arch | backbone          | loss |checkpoint  | CV   | test_a |
|:----|:-----|:------------------|:-----|:-----------|:-----|:-------|
|     | unet | resnet18          | bce  | best_loss  |      | 0.3093 |
| 3838| unet | resnet18          | bce  | best_loss  |0.4744| 0.4102 |
| 3850| unet | resnet18          | bce  | best_metric|0.4919|        |
| 3964| unet | resnet50          | bce  | best_metric|0.5544| 0.4975 |
| 3975| unet | se_resnext50_32x4d| bce  | best_metric|0.5644| 0.5118 |
| 4008| unet | se_resnext50_32x4d| lfs  | best_metric|0.6007| 0.5797 |
| **test** | **image** | **size changed from** | **256**  | **to 416** |
| 4010| unet | se_resnext50_32x4d| bce  | best_metric|0.6753| 0.6690 |
| 4132| unet3| se_resnext50_32x4d| bce  | swa        |0.6356| 0.6265 |
|     |      |                   |      | best_metric|0.6521|        |
| 4205| unet2_ds| se_resnext50   | bce  | swa        |0.6653|        |
| 4209| unet2| se_resnext101_32x4d| bce | swa        |0.6815| 0.6746 |
| 4256| unet2| senet154          | bce  | swa        |0.6892| 0.6869 
| 4298| unet2_e_scse| se_resnext101_32x4d | bce | swa, tta|0.69533|0.7017|
| 4334| unet2_e_scse| se_resnext101_32x4d | bce | swa, tta|0.73805|0.7319|

# references
* [Awesome Semantic Segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)
* [Resources of semantic segmantation based on Deep Learning model](https://github.com/tangzhenyu/SemanticSegmentation_DL)
* [Review of Deep Learning Algorithms for Image Semantic Segmentation](https://medium.com/@arthur_ouaknine/review-of-deep-learning-algorithms-for-image-semantic-segmentation-509a600f7b57)
* [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion)
* [Winning Solution for the Spacenet Challenge: Joint Learning with OpenStreetMap](https://i.ho.lc/winning-solution-for-the-spacenet-challenge-joint-learning-with-openstreetmap.html)