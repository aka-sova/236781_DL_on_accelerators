
from . import experiments as experiments



# to run this, i use the following command:
#   srun -c 2 --gres=gpu:1 --pty python -m hw2.run_depth_experiments

def run_experiments():

    print("Starting experiments 1")

    K_values_lists = [[32], [64]]
    L_values = [2, 4, 8, 16]

    for K in K_values_lists:
        for L in L_values:
            print(f"Starting: K = {K}, L = {L}")
            experiments.run_experiment(run_name=f'exp1_1_L{L}_K{K[0]}',
                                       bs_train=128,
                                       batches=150,
                                       epochs=70,
                                       early_stopping=5,
                                       filters_per_layer=K,
                                       layers_per_block=L,
                                       pool_every=4,
                                       hidden_dims=[1000],
                                       model_type='cnn',
                                       conv_params=dict(kernel_size=3, stride=1, padding=1),
                                       activation_type='lrelu',
                                       activation_params=dict(negative_slope=0.05))

    print("Starting experiments 2")

    K_values_lists = [[32], [64], [128], [256]]
    L_values = [2, 4, 8]

    for K in K_values_lists:
        for L in L_values:
            print(f"Starting: K = {K}, L = {L}")
            experiments.run_experiment(run_name=f'exp1_2_L{L}_K{K[0]}',
                                       bs_train=128,
                                       batches=150,
                                       epochs=70,
                                       early_stopping=5,
                                       filters_per_layer=K,
                                       layers_per_block=L,
                                       pool_every=4,
                                       hidden_dims=[1000],
                                       model_type='cnn',
                                       conv_params=dict(kernel_size=3, stride=1, padding=1),
                                       activation_type='lrelu',
                                       activation_params=dict(negative_slope=0.05))

    print("Starting experiments 3")

    K_values_lists = [[64, 128, 256]]
    L_values = [1, 2, 3, 4]

    for K in K_values_lists:
        K_str = [str(k_str) for k_str in K]
        K_str = '-'.join(K_str)
        for L in L_values:
            print(f"Starting: K = {K_str}, L = {L}")
            experiments.run_experiment(run_name=f'exp1_3_L{L}_K{K_str}',
                                       bs_train=128,
                                       batches=150,
                                       epochs=70,
                                       early_stopping=5,
                                       filters_per_layer=K,
                                       layers_per_block=L,
                                       pool_every=4,
                                       hidden_dims=[1000],
                                       model_type='cnn',
                                       conv_params=dict(kernel_size=3, stride=1, padding=1),
                                       activation_type='lrelu',
                                       activation_params=dict(negative_slope=0.05))

    print("Starting experiments 4-1")

    K_values_lists = [[32]]
    L_values = [8, 16, 32]

    for K in K_values_lists:
        K_str = [str(k_str) for k_str in K]
        K_str = '-'.join(K_str)
        for L in L_values:
            print(f"Starting: K = {K_str}, L = {L}")
            experiments.run_experiment(run_name=f'exp1_4_L{L}_K{K_str}',
                                       bs_train=128,
                                       batches=150,
                                       epochs=200,
                                       early_stopping=5,
                                       filters_per_layer=K,
                                       layers_per_block=L,
                                       pool_every=8,
                                       hidden_dims=[1000],
                                       model_type='resnet',
                                       conv_params=dict(kernel_size=3, stride=1, padding=1),
                                       activation_type='lrelu',
                                       activation_params=dict(negative_slope=0.05))

    print("Starting experiments 4-2")

    K_values_lists = [[64, 128, 256]]
    L_values = [2, 4, 8]

    for K in K_values_lists:
        K_str = [str(k_str) for k_str in K]
        K_str = '-'.join(K_str)
        for L in L_values:
            print(f"Starting: K = {K_str}, L = {L}")
            experiments.run_experiment(run_name=f'exp1_4_L{L}_K{K_str}',
                                       bs_train=128,
                                       batches=150,
                                       epochs=200,
                                       early_stopping=5,
                                       filters_per_layer=K,
                                       layers_per_block=L,
                                       pool_every=4,
                                       hidden_dims=[1000],
                                       model_type='resnet',
                                       conv_params=dict(kernel_size=3, stride=1, padding=1),
                                       activation_type='lrelu',
                                       activation_params=dict(negative_slope=0.05))


if __name__ == "__main__":
    run_experiments()