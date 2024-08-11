import csv, sys

with open("csv_outputs\quant_fp16_questions_and_responses_0-351.csv","r") as csv_in:
    csv_reader = csv.reader(csv_in)
    csv_out = open('q2qar_ref.csv',"w",newline="")
    csv_writer = csv.writer(csv_out)
    
    for lines in csv_reader:
        print(f"q_index: {lines[0]}")
        csv_writer.writerow([lines[0],lines[1].replace("\n",""),lines[2].lstrip().rstrip().replace("\n"," <br> ").lines[3]])
    
    csv_out.close()