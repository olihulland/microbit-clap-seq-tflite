import random

with open("data.csv", "r") as file:
    old_claps = []
    old_labels = []
    for line in file:
        asList = line.split(",")
        old_labels.append(1 if asList[-1].strip() == "true" else 0)
        old_claps.append(list(map(lambda numStr: int(numStr), asList[:-1])))

    MAX_TIME_BEFORE_TIMEOUT_MS = 3000
    MIN_TIME_BETWEEN_PRECEDING_AND_WAKE_MS = 500

    new_claps = []
    new_labels = []

    for i in range(len(old_claps)):
        working_claps = old_claps[i]

        """ add preceding claps """
        # add some precedinging claps before the wake sequence
        # random number of them but cannot be more than 10 total
        num_precedinging_claps = random.randint(0, 10 - len(working_claps)) # may want to consider weighting so that it is more likely to have 0 precedinging claps

        # shift the wake sequence to the right by the number of precedinging claps
        working_claps = ([-1] * num_precedinging_claps) + working_claps   # -1 is the padding value

        # fill with random times
        for j in range(0, num_precedinging_claps):
            working_claps[j] = random.randint(0, MAX_TIME_BEFORE_TIMEOUT_MS-200)

        # set the first to 0
        working_claps[0] = 0

        if num_precedinging_claps > 0:
            # set time between precedinging and wake
            time_between = random.randint(MIN_TIME_BETWEEN_PRECEDING_AND_WAKE_MS, MAX_TIME_BEFORE_TIMEOUT_MS-200)
            working_claps[num_precedinging_claps] = time_between

            new_claps.append(working_claps)
            new_labels.append(old_labels[i])

    # output to a file
    with open("data_more.csv", "w") as out:
        combined_claps = old_claps + new_claps
        combined_labels = old_labels + new_labels

        for i, claps in enumerate(combined_claps):
            out.write(f"{','.join(list(map(str,claps)))},{combined_labels[i]}\n")