from abc import ABC, abstractmethod


class EpisodeInfo(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class Timeout(EpisodeInfo):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Timeout'


class ReachGoal(EpisodeInfo):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Reaching goal'


class Danger(EpisodeInfo):
    def __init__(self, min_dist):
        super().__init__()
        self.min_dist = min_dist

    def __str__(self):
        return 'Too close'


class Collision(EpisodeInfo):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Collision'


class Nothing(EpisodeInfo):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return ''
