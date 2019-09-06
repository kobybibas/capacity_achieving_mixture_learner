import numpy as np
import pandas as pd
from tqdm import tqdm

from externals.blahut_arimoto_algorithm.blahut_arimoto_algorithm import blahut_arimoto
from externals.deep_pnml.src.analyze_utilities import compute_log_loss, entropy
from externals.deep_pnml.src.dataset_utilities import create_cifar10_dataloaders


def extract_genie_probabilities_list(evaluation_dict, label):
    # Extract to probabilities of the model which was trained with the true label
    # return list of probabilities
    true_label = evaluation_dict['true_label']
    prob_genie = np.array(evaluation_dict[str(label)]['prob'])
    predicted_genie_label = np.argmax(prob_genie)
    return prob_genie, true_label, predicted_genie_label


def result_dict_to_mixture_genie_df(results_dict: dict, genie_label: int):
    # Initialize columns to df
    df_col = [str(x) for x in range(10)] + ['true_label', 'loss', 'entropy']
    genie_dict = {}
    for col in df_col:
        genie_dict[col] = []
    loc = []

    # Iterate on keys
    for keys in results_dict:
        # extract probability of test sample
        sample_dict = results_dict[keys]
        prob_genie, true_label, predicted_genie_label = extract_genie_probabilities_list(
            sample_dict, genie_label)
        genie_dict['true_label'].append(true_label)
        genie_dict['loss'].append(compute_log_loss(prob_genie, true_label))
        for prob_label, prob_single in enumerate(prob_genie):
            genie_dict[str(prob_label)].append(prob_single)
        genie_dict['entropy'].append(entropy(prob_genie, base=10))
        loc.append(int(keys))

    # Create df
    genie_df = pd.DataFrame(genie_dict, index=loc)

    # Add more columns
    is_correct = np.array(genie_df[[str(x) for x in range(10)]].idxmax(axis=1)).astype(int) == np.array(
        genie_df['true_label']).astype(int)
    genie_df['is_correct'] = is_correct
    return genie_df


def genie_df_dict_to_mixture_df(genie_df_dict: dict, is_random: bool = False) -> pd.DataFrame:
    if is_random is True:
        # Import tesloader for random label experiment
        _, testloader, _ = create_cifar10_dataloaders('../data/', 1, 1)

    # Build B-A dataframe
    ref_df = genie_df_dict['0']

    mixture_df = pd.DataFrame(
        columns=[str(x) for x in range(10)] + ['loss', 'entropy', 'is_correct', 'capacity', 'true_label'],
        index=ref_df.index)
    for idx in tqdm(range(ref_df.shape[0])):

        prob_list = [df.iloc[idx][[str(x) for x in range(10)]].values for key, df in genie_df_dict.items()]
        prob = np.asarray(prob_list)
        prob[prob == 0] = 1e-17
        prob = prob.astype(float)
        # print()
        # print('Start')
        # print(prob)
        capacity_single, r_x = blahut_arimoto(prob, log_base=10)

        bla_ari_prob = np.zeros(prob.shape[1], dtype=float)
        # print('capacity_single, r_x: ', capacity_single, r_x)
        for r, prob_single in zip(r_x.astype(float), prob_list):
            bla_ari_prob += r * prob_single.astype(float)

        ref_row = ref_df.iloc[idx]
        idx_sample = ref_df.index.values[idx]
        true_label_single = ref_row['true_label'] if is_random is False else testloader.dataset.test_labels[
            int(idx_sample)]

        loss_single = compute_log_loss(bla_ari_prob, true_label_single)
        entropy_single = entropy(bla_ari_prob, base=10)
        is_correct_single = np.argmax(bla_ari_prob) == true_label_single

        # Add to row
        mixture_df.iloc[idx] = bla_ari_prob.tolist() + [loss_single,
                                                        entropy_single,
                                                        is_correct_single,
                                                        capacity_single,
                                                        true_label_single]

    # Verify dtype
    for x in range(10):
        mixture_df[str(x)] = mixture_df[str(x)].astype(float)
    mixture_df[['loss', 'entropy', 'capacity']] = mixture_df[['loss', 'entropy', 'capacity']].astype(float)
    mixture_df['true_label'] = mixture_df['true_label'].astype(int)
    mixture_df['is_correct'] = mixture_df['is_correct'].astype(bool)
    return mixture_df


def create_mixture_df(results_dict: dict, is_random: bool = False) -> pd.DataFrame:
    # Load genies
    genie_df_dict = {}
    for i in range(10):
        genie_df_dict[str(i)] = result_dict_to_mixture_genie_df(results_dict, i)

    mixture_df = genie_df_dict_to_mixture_df(genie_df_dict, is_random=is_random)
    return mixture_df


if __name__ == "__main__":
    e = 0.1
    p_1 = [1 - e, e, 0]
    p_2 = [0, e, 1 - e]
    p_y_x = np.asarray([p_1, p_2])
    print(p_y_x)
    print('    1')
    c_curr, r_x = blahut_arimoto(p_y_x, log_base=2)
    print(c_curr, r_x)
