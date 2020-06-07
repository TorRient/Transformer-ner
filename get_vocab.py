with open("word2vec/baomoi.window2.vn.model.txt", "r") as files:
    with open("vocab.txt", "w") as file_w:
        for idx, x in enumerate(files):
            if idx >= 1:
                file_w.write(x.split()[0])
                file_w.write("\n")