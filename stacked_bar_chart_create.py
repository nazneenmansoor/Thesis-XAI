# --------------------------------------------------------------------------------------------------------------------------
# stacked_bar_chart_create.py
# Description       : Module which creates stacked bar charts region-wise for three models
# Author            : Nazneen Mansoor
# Date              : 26/07/2023
# ---------------------------------------------------------------------------------------------------------------------------



import pandas as pd

from settings import get_settings
from matplotlib import pyplot as plt

def plot_hist():


    plt.figure(figsize=(12, 8))

    df = pd.DataFrame([['VGG16', 4, 0, 0, 12, 40, 32, 102, 76, 379], ['ResNet50', 43, 0, 129, 97, 125, 232, 256, 246, 950], ['InceptionV3', 15, 7, 10, 10, 24, 31, 47, 38, 266]],
                      columns=['Model', 'Eye', 'Cheek', 'Nose', 'Mouth', 'Neck', 'Cloth', 'Ear', 'Hair', 'Unlocalizable'])

    # plot data in stack manner of bar type
    df.plot(x='Model', kind='bar', stacked=True,
            title='Stacked Bar Graph Region-wise', width=0.5)

    plt.xticks(rotation = 0, ha='right')
    plt.ylabel("Number of interpretable units")
    plt.savefig('{}/Model_comparison.png'.format(opts.output_plots))
    plt.close()

if __name__ == '__main__':

    opts = get_settings()
    plot_hist()