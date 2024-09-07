import os
from itertools import islice

import pandas as pd
from matplotlib import (
    cm,
)
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

def get_plot_variables_distribution(plots_dir: str):
    """
    Get a function that can be used with applyInPandas to plot and save histograms of the data
    :param plots_dir:  directory where the plots will be saved
    :return:
    """
    plots_dir = os.path.abspath(plots_dir)
    os.makedirs(plots_dir, exist_ok=True)
    
    schema = StructType([
        StructField("variable", StringType(), False),
        StructField("file_path", StringType(), False),
    ])
    
    def plot_variable_distribution(
        variable_name: tuple[str],
        events: pd.DataFrame,
        variable_data: pd.DataFrame
    ) -> pd.DataFrame:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import re
        import pandas as pd
        import numpy as np
        
        variable_name = variable_name[0]
        
        # Determine the number of bins dynamically
        bin_edges = np.histogram_bin_edges(events['value'], bins='auto')
        if len(bin_edges) > 30:
            bin_edges = np.histogram_bin_edges(events['value'], bins=30)
        events['code'] = events['code'].fillna(0).astype(int)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        sns.histplot(
            data=events,
            x='value',
            hue='code',
            kde=False,
            color='skyblue',
            palette='colorblind',
            stat='density',
            multiple='stack',
            bins=bin_edges,
            ax=ax,
        )
        
        code_to_name = variable_data.set_index('code')['variable'].to_dict()
        legend = ax.get_legend()
        legend.set_loc('upper right')
        for text in legend.get_texts():
            if (code := int(text.get_text())) in code_to_name:
                text.set_text(code_to_name[code])
        
        unit = events['unit'].iloc[0]
        ax.set_xlabel(unit)
        
        fig.tight_layout()
        
        file_name = 'hist_' + re.sub(r"[^a-zA-Z0-9]", "_", variable_name) + ".eps"
        file_path = f'{plots_dir}/{file_name}'
        try:
            fig.savefig(file_path, dpi=300)
            fig.show()
        except Exception as e:
            print(f"Failed to save plot for group {variable_name}: {e}")
        finally:
            plt.close(fig)
        return pd.DataFrame(data=[[variable_name, file_path]])
    
    return plot_variable_distribution, schema


def get_plot_patient_journey(
    plots_dir: str,
    statistics: pd.DataFrame,
    /,
    file_formats: list[str] = None,
):
    if file_formats is None:
        file_formats = ['png', 'eps', 'svg']
    plots_dir = os.path.abspath(plots_dir)
    os.makedirs(plots_dir, exist_ok=True)
    
    schema = StructType([
        StructField("stay_id", IntegerType(), False),
        StructField("file_paths", ArrayType(StringType()), False),
    ])
    
    statistics = statistics.set_index('variable')[['p0.01', 'p0.99']]
    
    def plot_patient_journey(
        stay_id: tuple[int],
        patient_events: pd.DataFrame,
    ):
        import pandas as pd
        from collections import defaultdict
        
        from matplotlib import (
            patches as patches,
            pyplot as plt,
        )
        
        stay_id = stay_id[0]
        # count number of distinct source/name pairs
        variables_num = len(patient_events.groupby(['source', 'variable']))
        source_num = len(patient_events['source'].unique())
        inner_group_margin = 0.01
        figsize = (16, variables_num * 0.5 + 1 + (source_num - 1) * inner_group_margin)
        
        fig, axes = plt.subplots(
            variables_num,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={
                'hspace': 0, 'wspace': 0,
                'left': 0.15, 'right': 1 - 1 / figsize[0],
                'top': 0.98,
                'bottom': 0 + 0.5 / figsize[1] + inner_group_margin * (source_num - 1),
            })
        
        if variables_num == 1:
            axes = [axes]
        
        fig.tight_layout()
        
        patient_events = patient_events.astype({            'value': 'int16',        })
        
        # use relative time
        patient_events['time'] = patient_events['minute'] / 60
        max_time, min_time = patient_events['time'].max(), patient_events['time'].min()
        axes_iter = iter(axes)
        # Choose a colormap
        cmap = plt.cm.viridis
        
        ax_titles = defaultdict(list)
        for group_number, (source, source_group) in enumerate(patient_events.groupby('source')):
            group_size = source_group['variable'].nunique()
            group_axes = islice(axes_iter, group_size)
            
            for (variable_name, variable_events), ax in zip(
                source_group.groupby('variable'), group_axes, strict=True):
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0 - group_number * 0.01, pos.width, pos.height])
                
                # Remove borders around the axes
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                ax.set_yticks([])  # Remove y-ticks
                
                variable_events = variable_events.sort_values('time')
                
                # Normalize the values
                p1, p99 = statistics.loc[variable_name, ['p0.01', 'p0.99']]
                norm = plt.Normalize(p1, p99)
                
                shown_values = [variable_events.iloc[0]]
                minimized_values = []
                
                # iter rows starting from 1, add to shown_values if time difference is more than 100
                for _, row in variable_events.iterrows():
                    if row['time'] - shown_values[-1]['time'] > (max_time / 20):
                        shown_values.append(row)
                    else:
                        minimized_values.append(row)
                
                shown_values_df = pd.DataFrame(shown_values)
                minimized_values_df = pd.DataFrame(minimized_values)
                
                # timeline arrows
                ax.annotate('', xy=(min_time - (max_time - min_time) * 0.03, 0),
                            xytext=(max_time + (max_time - min_time) * 0.03, 0),
                            arrowprops=dict(arrowstyle="<-", lw=1, color='grey'), zorder=0)
                
                if minimized_values_df.shape[0] > 0:
                    ax.scatter(minimized_values_df['time'],
                               [0] * minimized_values_df.shape[0],
                               s=30,
                               alpha=0.5,
                               cmap=cmap,
                               c=minimized_values_df['value'],
                               norm=norm,
                               )
                
                # Plot circles
                ax.scatter(shown_values_df['time'], [0] * shown_values_df.shape[0], s=900,
                           cmap=cmap,
                           c=shown_values_df['value'],
                           norm=norm,
                           )
                # Annotate circles
                for _, row in shown_values_df.iterrows():
                    text_color = 'black' if norm(row['value']) > 0.5 else 'white'
                    ax.text(row['time'],
                            0,
                            str(row['value']),
                            ha='center',
                            va='center',
                            fontsize=12,
                            color=text_color, )
                
                # Set title on the left side
                ax_title = ax.text(-0.001,
                                   0.5,
                                   variable_name,
                                   transform=ax.transAxes,
                                   ha='right',
                                   va='center',
                                   fontsize=12,
                                   rotation=0)
                ax_titles[source].append(ax_title)
        
        margin = 0.1 / figsize[0], 0.1 / figsize[1]
        group_boxes = get_group_boxes(ax_titles, fig, margin)
        
        min_x0 = min([x0 for x0, _, _, _ in group_boxes.values()])
        max_x1 = max([x1 for _, _, x1, _ in group_boxes.values()])
        
        for source, box in group_boxes.items():
            y0, y1 = box[1], box[3]
            
            rect = patches.Rectangle(
                (min_x0, y0), max_x1 - min_x0, y1 - y0, transform=fig.transFigure,
                facecolor='none', edgecolor='black', linewidth=1, clip_on=False
            )
            fig.text((min_x0 + max_x1) / 2, y1, source.capitalize(),
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            fig.patches.append(rect)
        
        # Add colorbar
        cbar_ax = fig.add_axes((1 - 0.8 / figsize[0], 0.5 / figsize[1], 0.02,
                                1 - 0.85 / figsize[1]))  # [left, bottom, width, height]
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), cax=cbar_ax)
        ticks = cbar.get_ticks()
        cbar.set_ticks([ticks[0], ticks[-1]])  # Set the ticks to the first and last positions
        cbar.set_ticklabels(['Low', 'High'])  # Set custom tick labels
        
        # cbar.set_ticks([])  # Remove existing ticks
        
        axes[-1].set_xlabel('Hours from ICU admission')
        file_paths = []

        try:
            for file_format in file_formats:
                file_name = f"journey_{stay_id}.{file_format}"
                file_path = f'{plots_dir}/{file_name}'
                file_paths.append(file_path)
                fig.savefig(file_path)
            # fig.show()
        except Exception as e:
            print(f"Failed to save plot for group {stay_id}: {e}")
        finally:
            plt.close(fig)
        return pd.DataFrame(data=[[stay_id, file_paths]])
    
    return plot_patient_journey, schema


def get_group_boxes(ax_titles, fig, margin):
    group_boxes = {}
    for source, source_titles in ax_titles.items():
        bboxes = [title.get_window_extent() for title in source_titles]
        bboxes = [fig.transFigure.inverted().transform(bbox) for bbox in bboxes]
        
        x0 = min([bbox[0][0] for bbox in bboxes]) - margin[0]
        x1 = max([bbox[1][0] for bbox in bboxes]) + margin[0]
        
        y0 = min([bbox[0][1] for bbox in bboxes]) - margin[1]
        y1 = max([bbox[1][1] for bbox in bboxes]) + margin[1]
        
        group_boxes[source] = (x0, y0, x1, y1)
    return group_boxes



def pickle_args(variables, group_one, group_two):
    import pickle
    variable_name = variables[0]
    file_path = f'/tmp/args_{variable_name}.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump((variables, group_one, group_two), f)
    return pd.DataFrame(data=[[variable_name, file_path]], columns=['variable', 'file_path'])
