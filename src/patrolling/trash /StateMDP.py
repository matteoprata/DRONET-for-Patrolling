

from src.utilities.utilities import euclidean_distance, min_max_normalizer


class State:
    def __init__(self, aoi_idleness_ratio, time_distances, position, aoi_norm, time_norm,
                 position_norm, is_final, is_flying, objective):

        self._aoi_idleness_ratio: list = aoi_idleness_ratio
        self._time_distances: list = time_distances

        self.aoi_norm = aoi_norm
        self.time_norm = time_norm

        # UNUSED from here
        self._position = position
        self.is_final = is_final
        self._is_flying = is_flying
        self.position_norm = position_norm
        self._objective = objective

    def objective(self, normalized=True):
        return self._objective if not normalized else min_max_normalizer(self._objective, 0, self.position_norm+1)

    def is_flying(self, normalized=True):
        return self._is_flying if not normalized else int(self._is_flying)

    def aoi_idleness_ratio(self, normalized=True):
        return self._aoi_idleness_ratio if not normalized else min_max_normalizer(self._aoi_idleness_ratio, 0, self.aoi_norm)

    def time_distances(self, normalized=True):
        return self._time_distances if not normalized else min_max_normalizer(self._time_distances, 0, self.time_norm)

    def position(self, normalized=True):
        return self._position if not normalized else min_max_normalizer(self._position, 0, self.position_norm)

    def vector(self, normalized=True, rounded=False):
        """ NN INPUT """
        if not rounded:
            return list(self.aoi_idleness_ratio(normalized)) + list(self.time_distances(normalized))
        else:
            return [round(i, 2) for i in list(self.aoi_idleness_ratio(normalized))] + \
                   [round(i, 2) for i in list(self.time_distances(normalized))]

    def __repr__(self):
        return "res: {}\ndis: {}\n".format(self.aoi_idleness_ratio(), self.time_distances()) #self.is_flying(False), self.objective(False))

    @staticmethod
    def round_feature_vector(feature, rounding_digit):
        return [round(i, rounding_digit) for i in feature]
