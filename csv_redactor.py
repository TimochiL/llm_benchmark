import csv

types = (2,4,'fp16')
folder = "llama27bchat_csv"

for type in types:
    csv_path = f"{folder}\quant_{type}_questions_and_responses_0-389.csv"
    csv_out = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_out)
    
    csv_path = f"{folder}\quant_{type}_questions_and_responses_0-389_redacted.csv"
    csv_in = open(csv_path, "r")
    
    csv_writer.writerow(csv_in.readline().strip().split(","))
    
    csv_reader = csv.reader(csv_in)
    
    for line in csv_reader:
        csv_writer.writerow([line[0],line[1],line[2],'[REDACTED]',line[4]])
    
    csv_out.close()
    csv_in.close()