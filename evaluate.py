import csv

types = (2,4,'fp16')
folder = "llama27bchat_csv"

for type in types:
    csv_path = f"{folder}\quant_{type}_questions_and_responses_0-389_redacted.csv"
    csv_in = open(csv_path, "r")
    csv_in.readline()
    csv_reader = csv.reader(csv_in)
    
    num_correct = 0
    for line in csv_reader:
        if line[3] == '1':
            num_correct += 1

    csv_in.close()
    print(F"{type}: {num_correct}/390")