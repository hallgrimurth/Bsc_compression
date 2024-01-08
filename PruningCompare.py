from TrainingUtil import *
from model import *
from dataloader import *
from Pruner import *
from model import *


def Prune_Compare(dataset, range, epoch, finetune, train, r_train, attack, project_name, results_path):

    if dataset == "CIFAR10":
        data_loader = CIFAR10DataLoader()
        Ft = get_resnet18().to(device)
        Fr = get_resnet18().to(device)
    elif dataset == "FashionMNIST":
        data_loader = FASHIONMNISTDataLoader()
        Ft = TutorialNet().to(device)
        Fr = TutorialNet().to(device)

    train_loader = data_loader.train_loader
    test_loader = data_loader.test_loader
    classes = data_loader.get_classes()

    # Initial models
    # Ft = TutorialNet().to(device)
    # Fr = TutorialNet().to(device)

    # Ft = get_resnet18().to(device)
    # Fr = get_resnet18().to(device)
    if train:
        if dataset == "CIFAR10":
            train_and_save_model(Ft, train_loader, test_loader, attack, num_epochs=20, r_train=False, save_path="models/Ct.pt", project_name="CIFAR10")
            train_and_save_model(Fr, train_loader, test_loader, attack, num_epochs=20, r_train=True, save_path="models/Cr.pt", project_name="CIFAR10")
        elif dataset == "FashionMNIST":
            train_and_save_model(Ft, train_loader, test_loader, attack, num_epochs=20, r_train=False, save_path="models/Ft.pt", project_name="FashionMNIST")
            train_and_save_model(Fr, train_loader, test_loader, attack, num_epochs=20, r_train=True, save_path="models/Fr.pt", project_name="FashionMNIST")

    # #numpy array to save results
    results = np.zeros((9, 4))
    results_post = np.zeros((9, 4))


    # Loop for different sparsity levels
    for i in range(1, range):

        # Prune models
        config_list = [{
            'sparse_ratio': (0.1 * i),
            'op_types': ['Linear', 'Conv2d', 'BatchNorm2d'],
            'exclude_op_names': ['fc'],
        }]

        # Load models
        if dataset == "CIFAR10":
            Ft.load_state_dict(torch.load("models/Ct.pt"))
            Fr.load_state_dict(torch.load("models/Cr.pt"))

            Ft_pruner = ModelPruner(Ft, config_list, device, dummy_input=torch.rand((1, 3, 32, 32)))
            Fr_pruner = ModelPruner(Fr, config_list, device, dummy_input=torch.rand((1, 3, 32, 32)))

        elif dataset == "FashionMNIST":
            Ft.load_state_dict(torch.load("models/Ft.pt"))
            Fr.load_state_dict(torch.load("models/Fr.pt"))

            Ft_pruner = ModelPruner(Ft, config_list, device, dummy_input=torch.rand((1, 1, 28, 28)))
            Fr_pruner = ModelPruner(Fr, config_list, device, dummy_input=torch.rand((1, 1, 28, 28)))

        Ft_pruner.prune()
        Fr_pruner.prune()

        # Ft_test_error = epoch(test_loader, Ft)[0]
        # Fr_test_error = epoch(test_loader, Fr)[0]
        # Ft_adv_error = epoch_adversarial(test_loader, Ft, pgd_linf)[0]
        # Fr_adv_error = epoch_adversarial(test_loader, Fr, pgd_linf)[0]

        # results[i-1] = [Ft_test_error, Fr_test_error, Ft_adv_error, Fr_adv_error]

        # # Fine-tune models
        if finetune:
            if dataset == "CIFAR10":
                train_and_save_model(Ft, train_loader, test_loader, attack, num_epochs=epoch, r_train=r_train, save_path=f"models/Ct{i}.pt", project_name=project_name)
                train_and_save_model(Fr, train_loader, test_loader, attack, num_epochs=epoch, r_train=r_train, save_path=f"models/Cr{i}.pt", project_name=project_name)
            elif dataset == "FashionMNIST":
                train_and_save_model(Ft, train_loader, test_loader, attack, num_epochs=epoch, r_train=r_train, save_path=f"models/Ft{i}.pt", project_name=project_name)
                train_and_save_model(Fr, train_loader, test_loader, attack, num_epochs=epoch, r_train=r_train, save_path=f"models/Fr{i}.pt", project_name=project_name)

        # train_and_save_model(Ft, train_loader, test_loader, pgd_linf, num_epochs=epoch, r_train=r_train, save_path=f"models/Ct{i}.pt", project_name="Pruning_compare_normal_1")
        # train_and_save_model(Fr, train_loader, test_loader, pgd_linf, num_epochs=epoch, r_train=r_train, save_path=f"models/Cr{i}.pt", project_name="Pruning_compare_normal_1")


        # evaluate models and save results in a numpy array
        Ft_test_error = epoch(test_loader, Ft)[0]
        Fr_test_error = epoch(test_loader, Fr)[0]
        Ft_adv_error = epoch_adversarial(test_loader, Ft, pgd_linf)[0]
        Fr_adv_error = epoch_adversarial(test_loader, Fr, pgd_linf)[0]

        results_post[i-1] = [Ft_test_error, Fr_test_error, Ft_adv_error, Fr_adv_error]

    #     # Save results as csv
        # np.savetxt("fashion_results_1.csv", results, delimiter=",")
        np.savetxt(results_path, results_post, delimiter=",")

    return results_post
    
Prune_Compare("FashionMNIST", range=10, epoch=20, finetune=True, train=True, r_train=False, attack=pgd_linf, project_name="fashion_Pruning_compare_normal_1", results_path="fashion_tradisional_results_1.csv")



