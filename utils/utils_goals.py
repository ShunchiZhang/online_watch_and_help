
def convert_goal_spec(goal):
    """
    Convert the task goal into a format interpreted by the planner and model
    """
    goals = {}
    for key_count in goal:
        key = list(key_count.keys())[0]
        count = key_count[key]
        elements = key.split('_')

        predicate = "{}_{}_{}".format(elements[2], elements[1], elements[3])
        goals[predicate] = count

    return goals
