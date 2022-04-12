################################################################
# Collects and analyses information during an evolutionary run #
################################################################

from typing import Optional, List
from collections import OrderedDict


class Analyser:

    def __init__(self, verbosity: bool, **kwargs):

        self._verbosity: bool = verbosity
        self._pop_size: Optional[int] = kwargs.get('pop_size', None)
        self._maze_size: Optional[int] = kwargs.get('maze_size', None)

    def collect(self, results: List[dict]):

        # print(results)

        self._count_num_falls(results)

        self._calculate_final_state_distr(results)

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

        if self._verbosity:
            print(final_state_distr)
            print(final_state_distr_perc)

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
