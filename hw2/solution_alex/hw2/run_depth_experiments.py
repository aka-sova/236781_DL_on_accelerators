

from . import experiments as experiments


def run_experiments():

    print("Starting experiments")

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
                                       hidden_dims=[100]*2,
                                       model_type='cnn',
                                       conv_params=dict(kernel_size=3, stride=1, padding=1),
                                       activation_type='lrelu',
                                       activation_params=dict(negative_slope=0.05))

if __name__ == "__main__":
    run_experiments()