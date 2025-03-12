import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from ssp_topsis import SSP_TOPSIS

from pyrepo_mcda import normalizations as norms
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda import weighting_methods as mcda_weights


def main():

    # Load decision matrix with performance values
    df = pd.read_csv('dataset/data.csv', index_col='Technology')
    types = np.array([-1, 1, 1, 1, -1, 1, 1, -1, -1])
    matrix = df.to_numpy()
    old_matrix = copy.deepcopy(matrix)

    names = list(df.index)

    
    # analysis with criteria weights modification
    # Technical criteria

    results_pref = pd.DataFrame(index=names)
    results_rank = pd.DataFrame(index=names)

    
    # 5 technical criteria, 4 economic criteria
    w_tech_total = np.arange(0.1, 1.0, 0.1)
    w_econ_total = []

    weights_tab = np.zeros((1, 9))

    for wt in w_tech_total:
    
        we = 1 - wt
        w_econ_total.append(we)

        weights = np.zeros(9)
        # technical 5 criteria [0, 1, 2, 3, 4]
        weights[:5] = wt / 5
        # economic 4 criteria [5, 6, 7, 8]
        weights[5:] = we / 4

        weights_tab = np.concatenate((weights_tab, weights.reshape(1, -1)), axis = 0)

        # sustainability coefficient from matrix calculated based on standard deviation from normalized matrix
        n_matrix = norms.minmax_normalization(old_matrix, types)
        s = np.sqrt(np.sum(np.square(np.mean(n_matrix, axis = 0) - n_matrix), axis = 0) / n_matrix.shape[0])


        ssp_topsis = SSP_TOPSIS(normalization_method=norms.minmax_normalization)
        pref = ssp_topsis(matrix, weights, types, s_coeff=s)
        rank = rank_preferences(pref, reverse = True)

        results_pref[str(wt)] = pref
        results_rank[str(wt)] = rank


    results_pref = results_pref.rename_axis('Technology')
    results_rank = results_rank.rename_axis('Technology')
    results_pref.to_csv('./results/df_pref_weights_tech.csv')
    results_rank.to_csv('./results/df_rank_weights_tech.csv')

    weights_tab_df = pd.DataFrame(weights_tab)
    weights_tab_df = weights_tab_df.iloc[1:,:]
    weights_tab_df['Technical'] = w_tech_total
    weights_tab_df['Economical'] = w_econ_total
    weights_tab_df.to_csv('./results/weights_tech.csv')


    # plot results of analysis with criteria weights modification
    ticks = np.arange(1, 6)

    x1 = np.arange(0, len(w_tech_total))

    plt.figure(figsize = (8, 5))
    for i in range(results_rank.shape[0]):
        plt.plot(x1, results_rank.iloc[i, :], '*-', linewidth = 3)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(names[i], (x_max, results_rank.iloc[i, -1]),
                        fontsize = 16, #style='italic',
                        horizontalalignment='left')

    # plt.xlabel("Technical criteria importance rate " + r'$\rightarrow$' + "\n" + r'$\leftarrow$' + " Economic criteria importance rate", fontsize = 16)
    plt.xlabel("Technical criteria importance rate" , fontsize = 16)
    plt.ylabel("Rank", fontsize = 16)
    plt.xticks(x1, np.round(w_tech_total, 2), fontsize = 16)
    plt.yticks(ticks, fontsize = 16)
    plt.xlim(x_min - 0.2, x_max + 2)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = ':')
    plt.title('Floating photovoltaic systems rankings', fontsize = 16)
    plt.tight_layout()
    plt.savefig('./results/technology_rankings_weights_tech.png')
    plt.savefig('./results/technology_rankings_weights_tech.pdf')
    plt.show()




    # Economic

    results_pref = pd.DataFrame(index=names)
    results_rank = pd.DataFrame(index=names)

    
    # 5 technical criteria, 4 economic criteria

    w_econ_total = np.arange(0.1, 1.0, 0.1)
    w_tech_total = []

    weights_tab = np.zeros((1, 9))

    for we in w_econ_total:
    
        wt = 1 - we
        w_tech_total.append(wt)

        weights = np.zeros(9)
        # technical 5 [0, 1, 2, 3, 4]
        weights[:5] = wt / 5
        # economic 4 [5, 6, 7, 8]
        weights[5:] = we / 4

        weights_tab = np.concatenate((weights_tab, weights.reshape(1, -1)), axis = 0)

        # sustainability coefficient from matrix calculated based on standard deviation from normalized matrix
        n_matrix = norms.minmax_normalization(old_matrix, types)
        s = np.sqrt(np.sum(np.square(np.mean(n_matrix, axis = 0) - n_matrix), axis = 0) / n_matrix.shape[0])


        ssp_topsis = SSP_TOPSIS(normalization_method=norms.minmax_normalization)
        pref = ssp_topsis(matrix, weights, types, s_coeff=s)
        rank = rank_preferences(pref, reverse = True)

        results_pref[str(we)] = pref
        results_rank[str(we)] = rank


    results_pref = results_pref.rename_axis('Technology')
    results_rank = results_rank.rename_axis('Technology')
    results_pref.to_csv('./results/df_pref_weights_eco.csv')
    results_rank.to_csv('./results/df_rank_weights_eco.csv')

    weights_tab_df = pd.DataFrame(weights_tab)
    weights_tab_df = weights_tab_df.iloc[1:,:]
    weights_tab_df['Technical'] = w_tech_total
    weights_tab_df['Economical'] = w_econ_total
    weights_tab_df.to_csv('./results/weights_eco.csv')


    # plot results of analysis with criteria weights modification
    ticks = np.arange(1, 6)

    x1 = np.arange(0, len(w_econ_total))

    plt.figure(figsize = (8, 5))
    for i in range(results_rank.shape[0]):
        plt.plot(x1, results_rank.iloc[i, :], '*-', linewidth = 3)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(names[i], (x_max, results_rank.iloc[i, -1]),
                        fontsize = 16, #style='italic',
                        horizontalalignment='left')

    # plt.xlabel("Technical criteria importance rate " + r'$\rightarrow$' + "\n" + r'$\leftarrow$' + " Economic criteria importance rate", fontsize = 16)
    plt.xlabel("Economic criteria importance rate", fontsize = 16)
    plt.ylabel("Rank", fontsize = 16)
    plt.xticks(x1, np.round(w_econ_total, 2), fontsize = 16)
    plt.yticks(ticks, fontsize = 16)
    plt.xlim(x_min - 0.2, x_max + 2)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = ':')
    plt.title('Floating photovoltaic systems rankings', fontsize = 16)
    plt.tight_layout()
    plt.savefig('./results/technology_rankings_weights_eco.png')
    plt.savefig('./results/technology_rankings_weights_eco.pdf')
    plt.show()
    

    # analysis with sustainability coefficient modification

    # All criteria
    results_pref = pd.DataFrame(index=names)
    results_rank = pd.DataFrame(index=names)


    sust_coeff = np.arange(0, 1.1, 0.1)
    new_sust_coeff = np.arange(0, 1.1, 0.05)

    for sc in sust_coeff:

        weights = mcda_weights.equal_weighting(matrix)

        s = np.ones(9) * sc

        ssp_topsis = SSP_TOPSIS(normalization_method=norms.minmax_normalization)
        pref = ssp_topsis(matrix, weights, types, s_coeff=s)
        rank = rank_preferences(pref, reverse = True)

        results_pref[str(sc)] = pref
        results_rank[str(sc)] = rank


    results_pref = results_pref.rename_axis('Technology')
    results_rank = results_rank.rename_axis('Technology')
    results_pref.to_csv('./results/df_pref_sust_all.csv')
    results_rank.to_csv('./results/df_rank_sust_all.csv')

    # plot results of analysis with sustainabiblity coefficient modification
    ticks = np.arange(1, 6)

    x1 = np.arange(0, len(sust_coeff))
    new_x1 = np.arange(0, len(new_sust_coeff))

    plt.figure(figsize = (8, 5))
    for i in range(results_rank.shape[0]):
        plt.plot(x1, results_rank.iloc[i, :], '*-', linewidth = 3)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(names[i], (x_max, results_rank.iloc[i, -1]),
                        fontsize = 16, #style='italic',
                        horizontalalignment='left')

    plt.xlabel("Sustainability coeffcient", fontsize = 16)
    plt.ylabel("Rank", fontsize = 16)
    
    # plt.xticks(np.linspace(x_min, x_max, len(sust_coeff)), np.round(sust_coeff, 2), fontsize = 16)
    # ax.set_xticks(np.arange(min(new_x1), max(new_x1), 0.1), fontsize = 16)
    ax.set_xticks(x1, np.round(sust_coeff, 2), fontsize = 16)
    plt.yticks(ticks, fontsize = 16)
    plt.xlim(x_min - 0.2, x_max + 2.4)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = ':')
    plt.title('Floating photovoltaic systems rankings', fontsize = 16)
    plt.tight_layout()
    plt.savefig('./results/rankings_sust_all.png')
    plt.savefig('./results/rankings_sust_all.pdf')
    plt.show()


    # Technical criteria
    results_pref = pd.DataFrame(index=names)
    results_rank = pd.DataFrame(index=names)


    sust_coeff = np.arange(0, 1.1, 0.1)

    for sc in sust_coeff:

        weights = mcda_weights.equal_weighting(matrix)

        s = np.zeros(9)
        s[[0,1,2,3,4]] = sc

        ssp_topsis = SSP_TOPSIS(normalization_method=norms.minmax_normalization)
        pref = ssp_topsis(matrix, weights, types, s_coeff=s)
        rank = rank_preferences(pref, reverse = True)

        results_pref[str(sc)] = pref
        results_rank[str(sc)] = rank


    results_pref = results_pref.rename_axis('Technology')
    results_rank = results_rank.rename_axis('Technology')
    results_pref.to_csv('./results/df_pref_sust_tech.csv')
    results_rank.to_csv('./results/df_rank_sust_tech.csv')

    # plot results of analysis with sustainabiblity coefficient modification
    ticks = np.arange(1, 6)

    x1 = np.arange(0, len(sust_coeff))

    plt.figure(figsize = (8, 5))
    for i in range(results_rank.shape[0]):
        plt.plot(x1, results_rank.iloc[i, :], '*-', linewidth = 3)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(names[i], (x_max, results_rank.iloc[i, -1]),
                        fontsize = 16, #style='italic',
                        horizontalalignment='left')

    plt.xlabel("Sustainability coeffcient", fontsize = 16)
    plt.ylabel("Rank", fontsize = 16)
    plt.xticks(x1, np.round(sust_coeff, 2), fontsize = 16)
    plt.yticks(ticks, fontsize = 16)
    plt.xlim(x_min - 0.2, x_max + 2.4)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = ':')
    plt.title('Floating photovoltaic systems rankings', fontsize = 16)
    plt.tight_layout()
    plt.savefig('./results/technology_rankings_sust_tech.png')
    plt.savefig('./results/technology_rankings_sust_tech.pdf')
    plt.show()



    # Economic criteria
    results_pref = pd.DataFrame(index=names)
    results_rank = pd.DataFrame(index=names)


    sust_coeff = np.arange(0, 1.1, 0.1)

    for sc in sust_coeff:

        weights = mcda_weights.equal_weighting(matrix)

        s = np.zeros(9)
        s[[5,6,7,8]] = sc

        ssp_topsis = SSP_TOPSIS(normalization_method=norms.minmax_normalization)
        pref = ssp_topsis(matrix, weights, types, s_coeff=s)
        rank = rank_preferences(pref, reverse = True)

        results_pref[str(sc)] = pref
        results_rank[str(sc)] = rank


    results_pref = results_pref.rename_axis('Technology')
    results_rank = results_rank.rename_axis('Technology')
    results_pref.to_csv('./results/df_pref_sust_eco.csv')
    results_rank.to_csv('./results/df_rank_sust_eco.csv')

    # plot results of analysis with sustainabiblity coefficient modification
    ticks = np.arange(1, 6)

    x1 = np.arange(0, len(sust_coeff))

    plt.figure(figsize = (8, 5))
    for i in range(results_rank.shape[0]):
        plt.plot(x1, results_rank.iloc[i, :], '*-', linewidth = 3)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(names[i], (x_max, results_rank.iloc[i, -1]),
                        fontsize = 16, #style='italic',
                        horizontalalignment='left')

    plt.xlabel("Sustainability coeffcient", fontsize = 16)
    plt.ylabel("Rank", fontsize = 16)
    plt.xticks(x1, np.round(sust_coeff, 2), fontsize = 16)
    plt.yticks(ticks, fontsize = 16)
    plt.xlim(x_min - 0.2, x_max + 2.4)
    plt.gca().invert_yaxis()
    
    plt.grid(True, linestyle = ':')
    plt.title('Floating photovoltaic systems rankings', fontsize = 16)
    plt.tight_layout()
    plt.savefig('./results/technology_rankings_sust_eco.png')
    plt.savefig('./results/technology_rankings_sust_eco.pdf')
    plt.show()



if __name__ == '__main__':
    main()