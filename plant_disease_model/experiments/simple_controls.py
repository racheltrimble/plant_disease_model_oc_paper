
def no_control(target_settings):
    target_settings.controller = {"type": "fixed_control",
                                  "settings": {"action": "no_action"}}
    target_settings.run_limits = {"tuning_valid": False,
                                  "training_valid": False,
                                  "multiple_evals": False}
    target_settings.set_display_name_addendum("no control")
    return target_settings


def equal_control(target_settings):
    target_settings.controller = {"type": "fixed_control",
                                  "settings": {"action": "equal"}}
    target_settings.run_limits = {"tuning_valid": False,
                                  "training_valid": False,
                                  "multiple_evals": False}
    target_settings.set_display_name_addendum("equal control")
    return target_settings


def equal_per_host_control(target_settings):
    target_settings.controller = {"type": "fixed_control",
                                  "settings": {"action": "equal_per_host"}}
    target_settings.run_limits = {"tuning_valid": False,
                                  "training_valid": False,
                                  "multiple_evals": False}
    target_settings.set_display_name_addendum("equal per host control")
    return target_settings


def snap_to_control_fully_obs(target_settings):
    target_settings.controller = {"type": "snap_to_control",
                                  "settings": {"observability": "full"}}
    target_settings.run_limits = {"tuning_valid": False,
                                  "training_valid": False,
                                  "multiple_evals": False}
    target_settings.set_display_name_addendum("snap to control")
    return target_settings


simple_controls = [no_control, equal_per_host_control]


def baseline(target_settings):
    target_settings.set_display_name_addendum("baseline")
    return target_settings
