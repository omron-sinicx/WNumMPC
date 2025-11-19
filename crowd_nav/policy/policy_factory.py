policy_factory = dict()  # dicts of policy classes


def none_policy():
    return None


from crowd_nav.policy.orca import ORCA
from crowd_nav.policy.social_force import SOCIAL_FORCE
from crowd_nav.policy.wnum_mpc import WNumMPC
from crowd_nav.policy.cadrl import CADRL

policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
policy_factory['social_force'] = SOCIAL_FORCE
policy_factory['wnum_mpc'] = WNumMPC
policy_factory["vanilla_mpc"] = WNumMPC
policy_factory["mean_mpc"] = WNumMPC
policy_factory['cadrl'] = CADRL

