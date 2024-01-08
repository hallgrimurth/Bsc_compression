from TrainingUtil import *
from model import *
from dataloader import *
from Pruner import *


# Load the dataset
data_loader = FASHIONMNISTDataLoader()
train_loader = data_loader.train_loader
test_loader = data_loader.test_loader
classes = data_loader.get_classes()

#Pruner config
pruning_config_list = [{
    'sparse_ratio': 0.5,
    'op_types': ['Linear', 'Conv2d'],
    'exclude_op_names': ['fc2'],
}]

# #Quantizer config
# bit_width = 'int8'
# quantizer_config_list = [{
#     'op_names': ['conv1','conv2',
#                  'conv3',
#                  'conv4','fc1',
#                  'fc2'],
#     'target_names': ['_input_', 'weight', '_output_'],
#     'quant_dtype': bit_width,
#     'quant_scheme': 'affine',
#     'granularity': 'default'
#     }]


Ft   = TutorialNet().to(device)
Fr   = TutorialNet().to(device)

# Ftp  = TutorialNet().to(device)
# Frp  = TutorialNet().to(device)
# Fftp = TutorialNet().to(device)
# Ffrp = TutorialNet().to(device)
# Ftpr = TutorialNet().to(device)

# Ftq  = TutorialNet().to(device)
# Frq  = TutorialNet().to(device)


train_and_save_model(Ft, train_loader, test_loader, pgd_linf, num_epochs=20, r_train=False, save_path="Ft128.pt", project_name="Fashion128")
train_and_save_model(Fr, train_loader, test_loader, pgd_linf, num_epochs=20, r_train=True, save_path="Fr128.pt", project_name="Fashion128")

# numpy array to save results
# results = np.zeros((9, 4))
# results_post = np.zeros((9, 4))

# # Loop for different sparsity levels
# for i in range(1, 10):

#     # Load models
#     Ft = TutorialNet().to(device)
#     Fr = TutorialNet().to(device)

#     # Prune models
#     config_list = [{
#         'sparse_ratio': (0.01 * i + 0.9),
#         'op_types': ['Linear', 'Conv2d'],
#         'exclude_op_names': ['fc2'],
#     }]
#     Ft_pruner = ModelPruner(Ft, config_list, device)
#     Fr_pruner = ModelPruner(Fr, config_list, device)

#     Ft.load_state_dict(torch.load("models/Ft.pt"))
#     Fr.load_state_dict(torch.load("models/Fr.pt"))

#     Ft_pruner.prune()
#     Fr_pruner.prune()

#     # pre_ft_test_error = epoch(test_loader, Ft)[0]
#     # pre_ft_adv_error = epoch_adversarial(test_loader, Ft, pgd_linf)[0]
#     # pre_fr_test_error = epoch(test_loader, Fr)[0]
#     # pre_fr_adv_error = epoch_adversarial(test_loader, Fr, pgd_linf)[0]

#     # results[i-1, 0] = pre_ft_test_error
#     # results[i-1, 1] = pre_ft_adv_error
#     # results[i-1, 2] = pre_fr_test_error
#     # results[i-1, 3] = pre_fr_adv_error


#     train_and_save_model(Ft, train_loader, test_loader, pgd_linf, num_epochs=3, r_train=True, save_path=f"models/Ft128_{i * 0.01 + 0.9}.pt", project_name="Fashion128Pruned90")
#     train_and_save_model(Ft, train_loader, test_loader, pgd_linf, num_epochs=3, r_train=True, save_path=f"models/Ft128R_{i* 0.01 + 0.9}.pt", project_name="Fashion128Pruned90")


    # Ft_test_error = epoch(test_loader, Ft)[0]
    # Fr_test_error = epoch(test_loader, Fr)[0]
    # Ft_adv_error = epoch_adversarial(test_loader, Ft, pgd_linf)[0]
    # Fr_adv_error = epoch_adversarial(test_loader, Fr, pgd_linf)[0]

    # results_post[i-1, 0] = Ft_test_error
    # results_post[i-1, 1] = Ft_adv_error
    # results_post[i-1, 2] = Fr_test_error
    # results_post[i-1, 3] = Fr_adv_error

#save results to csv
    # np.savetxt("results3.csv", results, delimiter=",")
    # np.savetxt("results3_post.csv", results_post, delimiter=",")
    



# config_list = [{
#         'sparse_ratio': (0.6),
#         'op_types': ['Linear', 'Conv2d'],
#         'exclude_op_names': ['fc2'],
#   }]
# Ft.load_state_dict(torch.load("models/Ft.pt"))

# Fr_pruner = ModelPruner(Ft, config_list, device)
# Fr_pruner.pytorch_prune(amount=0.6)

# train_and_save_model(Ft, train_loader, test_loader, pgd_linf, num_epochs=40, r_train=True, save_path="models/Ft->R.pt", project_name="Fashion128Pruned")

# print("Fr test_error: ", epoch(test_loader, Fr))
# print("Fr adv_error: ", epoch_adversarial(test_loader, Fr, pgd_linf))
# print("Fcr test_error: ", epoch(test_loader, Ft))
# print("Fcr adv_error: ", epoch_adversarial(test_loader, Ft, pgd_linf))



# '''
# print("Evaluating the models")
# print("Test error | Adversarial error")
# print("Ft: ", terr_ft, pgd_ft)
# print("Fr: ", terr_fr, pgd_fr)


# '''
# print("Ftp: ", terr_ftp, pgd_ftp)
# print("Frp: ", terr_frp, pgd_frp)
# print("Fftp: ", terr_fftp, pgd_fftp)
# print("Ffrp: ", terr_ffrp, pgd_ffrp)
# print("Ftpr: ", terr_ftpr, pgd_tpr)

