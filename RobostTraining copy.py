from TrainingUtil import *
from model import *
from dataloader import *
from Pruner import *


# Load the dataset
data_loader = FASHIONMNISTDataLoader()
train_loader = data_loader.train_loader
test_loader = data_loader.test_loader
classes = data_loader.get_classes()

# Initial models
Ft = TutorialNet().to(device)
Fr = TutorialNet().to(device)
# Ftp = TutorialNet().to(device)
# Frp = TutorialNet().to(device)

# Training without pruning
train_and_save_model(Ft, train_loader, test_loader, pgd_linf, num_epochs=40, r_train=False, save_path="models/Ft.pt", project_name="FashionMNIST_cleaned")
train_and_save_model(Fr, train_loader, test_loader, pgd_linf, num_epochs=40, r_train=True, save_path="models/Fr.pt", project_name="FashionMNIST_cleaned")

# def prune_and_load(model, sparsity, save_path, device):
#     config_list = [{
#         'sparse_ratio': sparsity,
#         'op_types': ['Linear', 'Conv2d'],
#         'exclude_op_names': ['fc2'],
#     }]
#     model.load_state_dict(torch.load(save_path))
#     pruner = ModelPruner(model, config_list, device)
#     pruner.prune()
    


# #numpy array to save results
# results = np.zeros((9, 4))

# # Loop for different sparsity levels
# for i in range(1, 10):

#     # Load models
#     Ft = TutorialNet().to(device)
#     Fr = TutorialNet().to(device)

#     # Prune models
#     config_list = [{
#         'sparse_ratio': (0.1 * i),
#         'op_types': ['Linear', 'Conv2d'],
#         'exclude_op_names': ['fc2'],
#     }]
#     Ft_pruner = ModelPruner(Ft, config_list, device)
#     Fr_pruner = ModelPruner(Fr, config_list, device)

#     Ft_pruner.pytorch_prune
#     Fr_pruner.pytorch_prune

#     Ft.load_state_dict(torch.load("models/Ft.pt"))
#     Fr.load_state_dict(torch.load("models/Fr.pt"))


#     # print("Ft test_error: ", epoch(test_loader, Ft))
#     # print("Fr test_error: ", epoch(test_loader, Fr))
#     # print("Ft adv_error: ", epoch_adversarial(test_loader, Ft, pgd_linf))
#     # print("Fr adv_error: ", epoch_adversarial(test_loader, Fr, pgd_linf))

#     # Fine-tune models
#     train_and_save_model(Ft, train_loader, test_loader, pgd_linf, num_epochs=5, r_train=False, save_path=f"models/NFTft{i}_2.pt", project_name="PytorchFashionMNIST")
#     train_and_save_model(Fr, train_loader, test_loader, pgd_linf, num_epochs=5, r_train=False, save_path=f"models/NFTfr_{i}_2.pt", project_name="PytorchFashionMNIST")

#       # Load models
#     # prune_and_load(Ft, 0.1 * i, f"models/Ft.pt", device)
#     # prune_and_load(Fr, 0.1 * i, f"models/Fr.pt", device)

#     #evaluate models and save results in a numpy array
#     Ft_test_error = epoch(test_loader, Ft)[0]
#     Fr_test_error = epoch(test_loader, Fr)[0]
#     Ft_adv_error = epoch_adversarial(test_loader, Ft, pgd_linf)[0]
#     Fr_adv_error = epoch_adversarial(test_loader, Fr, pgd_linf)[0]

#     # Save results
#     results[i-1] = [Ft_test_error, Fr_test_error, Ft_adv_error, Fr_adv_error]

#     # Print array
#     print(results)

#     # Save array
#     np.save("NFT_results_2.npy", results)


