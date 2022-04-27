################################################################
# Collects and analyses information during an evolutionary run #
################################################################

from typing import Optional, List
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from heatmap_plot import heatmap, annotate_heatmap


class Analyser:

    def __init__(self, verbosity: bool, **kwargs):

        self._verbosity: bool = verbosity
        self._pop_size: Optional[int] = kwargs.get('pop_size', None)
        self._maze_width: Optional[int] = kwargs.get('maze_width', None)
        self._maze_height: Optional[int] = kwargs.get('maze_height', None)
        self._maze_size: Optional[int] = self._maze_width * self._maze_height
        self._maze = kwargs.get('maze', None)

        # Format maze
        desc = kwargs.get('maze', None)
        self._maze = []
        for m_row in desc:
            self._maze.append([cha for cha in m_row])

    def collect(self, results: List[dict], final_gen: bool):

        # print(results)

        self._count_num_falls(results)

        state_array = self._calculate_final_state_distr(results)
        if final_gen:
            self._plot_final_state_distr(state_array)

    def _calculate_final_state_distr(self, results: List[dict]):

        final_state_distr = OrderedDict()
        for i in range(self._maze_size):
            final_state_distr[str(i)] = 0

        num_trials = 0

        for indv_res in results:
            trial_num = 0
            while True:
                try:
                    final_state = str(indv_res[str(trial_num)]['final_state'])
                    final_state_distr[final_state] += 1
                    num_trials += 1
                except KeyError:
                    break
                trial_num += 1

        # Convert to percentages
        final_state_distr_perc = final_state_distr.copy()
        for k, v in sorted(final_state_distr_perc.items()):
            final_state_distr_perc[k] = v / num_trials

        #if self._verbosity:
        #    print(final_state_distr)
        #    print(final_state_distr_perc)

        # Convert from dictionary to 2d numpy array
        state_array = self._convert_final_state_distr_to_array(final_state_distr_perc)
        print(state_array)

        return state_array

    # Plot heat map of final state distribution
    def _plot_final_state_distr(self, state_array):

        fig, ax = plt.subplots()
        im, cbar = heatmap(state_array, None, None, ax=ax,
                           # cmap="YlGn", cbarlabel="Final state percentage")
                           #cmap='jet',
                           cbarlabel="Final state percentage", vmin=0., vmax=1.)

        # Prepare and plot heat map text
        hm_text = self._maze.copy()
        for i in range(len(hm_text)):
            for j in range(len(hm_text[0])):
                coords = '\n(' + str(j) + ', ' + str(i) + ')\n'
                new_text = hm_text[i][j] + coords + str(state_array[i, j])
                hm_text[i][j] = new_text
        texts = annotate_heatmap(im, text_data=hm_text, colour_data=state_array,
                                 threshold=0.5)

        fig.tight_layout()
        plt.axis('off')
        plt.savefig('figures/hm_figure.png', bbox_inches='tight',
                    pad_inches=0.05, dpi=500)
        plt.show()

    def _convert_final_state_distr_to_array(self, final_state_distr):

        final_state_array = []
        for i in range(self._maze_height):
            final_state_row = []
            for j in range(self._maze_width):
                state_int = self._maze_width * i + j
                final_state_row.append(final_state_distr[str(state_int)])
            final_state_array.append(final_state_row)

        return np.array(final_state_array)

    # Count total number of falls in the generation
    def _count_num_falls(self, results: List[dict]):

        num_falls = 0

        for indv_res in results:
            trial_num = 0
            while True:
                try:
                    if indv_res[str(trial_num)]['hole']:
                        num_falls += 1
                except KeyError:
                    break
                trial_num += 1

        if self._verbosity:
            print('Number of falls:', num_falls)
            print('% num falls:', num_falls / self._pop_size)
