"A CNN-GRU Neural Network Framework based on Bayesian Optimization of
Hyperparameters used to Predict Tetramer Protein-Protein Interaction"
Data files:
Tetramer Protein-Protein:
train_20_set.csv and train_labels.txt
val_20_set.csv and val_labels.txt
test_20_set.csv and test_labels.txt
Code files:
CNN-GRU.py    train the model "CNN-GRU Based on Bayesian Optimization of Hyperparameters"
draw_bar.py   draw the contrast figure
holdonAUC.py  draw the ROC curve figure
load_model.py test the model
read_loss.py  draw the loss figure
data_augmentation.py data augmentation
Usage:
train:
     python CNN_GRU.py
test:
     python load_model.py
Requirements:
python3.8
pytorch1.9.0+cu102
scikit-learn0.23
torchvision0.10.0+cpu
numpy1.20.1
pandas1.2.4
matplotlib3.3.4
enum34-1.1.10
pandas-ml-0.6.1
imbalanced-learn-0.8.1
joblib(>=0.11)
Reference:
[1] Wardah W ,  Dehzangi A ,  Taherzadeh G , et al. Predicting protein-peptide binding sites with a Deep Convolutional
Neural Network[J]. Journal of Theoretical Biology, 2020, 496(1):110278.
[2]Zhe, Wang, Chunhua, et al. SMOTE-Tomek-Based
Resampling for Personality Recognition[J].IEEE Access, 2019, 7:129678-129689.
[3]Chua L ,  Roska T ,  Kozek T , et al. The CNN paradigm - a short tutorial[J].  1993.
[4]Y  Liu,  Qin H ,  Zhang Z , et al. Ensemble spatiotemporal forecasting of solar irradiation using variational
Bayesian convolutional gate recurrent unit network[J]. Applied Energy, 2019, 253(1):113596.
[5] Patel S ,  Tripathi R ,  Kumari V , et al. DeepInteract: Deep Neural Network based Protein-Protein Interaction
prediction tool[J]. Current Bioinformatics, 2017.
[6]Gustafsson O ,  Villani M ,  Stockhammar P . Bayesian Optimization of Hyperparameters when the Marginal Likelihood
is Estimated by MCMC[J]. Papers, 2020.
Please cite the associated publication when using these codes.

