def read_mail(path):
    return open(path, 'r', errors='ignore').read()

def read_mails(dir, amount=''):
    import os
    documents = []

    files = [os.path.join(dir, file) for file in os.listdir(dir)]

    if not amount:
        amount = len(files)
    else:
        amount = int(amount)

    print("Reading "+str(amount)+" files")

    for i, file in enumerate(files):
        doc = open(file, "r", errors="ignore").read()
        documents.append(doc)

        if i == amount:
            break

    return list(set(documents))


def save_to_file(path, data, mode='wb'):
    import pickle

    with open(path, mode) as f:
        pickle.dump(data, f)
        f.close()

    return 0


def load_from_file(path, mode='rb'):
    import pickle

    with open(path, mode) as f:
        data = pickle.load(f)
        f.close()
        return data

    return 0
