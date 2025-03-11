# --------------------------------------------------------------------------------------------------------------------------
# stacked_bar_chart_create.py
# Description       : Module which creates stacked bar charts for different layers in each model
# Author            : Nazneen Mansoor
# Date              : 27/07/2023
# ---------------------------------------------------------------------------------------------------------------------------

import pandas as pd

from settings import get_settings
from matplotlib import pyplot as plt

def plot_hist():


    plt.figure(figsize=(12, 8))

    df = pd.DataFrame([['Conv1', 47, 17], ['Conv2', 70, 26], ['Conv3', 152, 40],  ['Conv4', 158, 34], ['Conv5', 350, 98]],
                      columns=['Layers', 'Interpretable', 'UnInterpretable'])
    print(df)

    # plot data in stack manner of bar type
    ax = df.plot(x='Layers', kind='bar', stacked=True,
            title='Units layerwise comparison - InceptionV3', width=0.3)
    for c in ax.containers:
        # Optional: if the segment is small or 0, customize the labels
        labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in c]

        # remove the labels parameter if it's not needed for customized labels
        ax.bar_label(c, labels=labels, label_type='center')

    plt.xticks(rotation=0, ha='right')

    plt.xlabel("Layer names")
    plt.ylabel("Number of units")
    plt.savefig('{}/Layer-comparison/Inception-V3.png'.format(opts.output_plots))
    plt.close()


if __name__ == '__main__':

    opts = get_settings()
    plot_hist()