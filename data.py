import re


class Data:
    data = []
    target_3 = []
    target_4 = []
    target_3_labels = []
    target_4_labels = []


def load_data():
    data = Data()
    folders = ["Dennis+Schwartz", "James+Berardinelli", "Scott+Renshaw", "Steve+Rhodes"]
    data.target_3_labels = ["r<=0.4", "0.4<r<0.7", "0.7<=r"]
    data.target_4_labels = ["r<=0.3", "0.4<=r<=0.5", "0.6<=r<=0.7", "0.8<=r"]

    for name in folders:
        file = open(f"scaledata/{name}/subj.{name}", "r")
        lines = file.readlines()
        for line in lines:
            data.data.append(re.findall(r'(\w[\w\']*)', line))
        file.close()

        file = open(f"scaledata/{name}/label.3class.{name}", "r")
        lines = file.readlines()
        for line in lines:
            data.target_3.append(int(line.strip()))
        file.close()

        file = open(f"scaledata/{name}/label.4class.{name}", "r")
        lines = file.readlines()
        for line in lines:
            data.target_4.append(int(line.strip()))
        file.close()

    return data
