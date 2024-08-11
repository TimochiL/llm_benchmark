import csv, math

types = (2,4,'fp16')
folder = "llama27bchat_csv"
csv_type = ""

for type in types:
    csv_path = f"{folder}\quant_{type}_questions_and_responses_0-389_categorized{csv_type}.csv"
    csv_out = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_out)
    
    csv_path = f"{folder}\quant_{type}_questions_and_responses_0-389{csv_type}.csv"
    csv_in = open(csv_path, "r")
    
    csv_in.readline()
    
    csv_reader = csv.reader(csv_in)
    
    labels = ['q_index','q_category','question','response','pass']
    csv_writer.writerow(labels)
    
    categories = ('Illegal Activity', 'Hate Speech', 'Malware', 'Physical Harm', 'Economic Harm', 'Fraud', 'Pornography', 'Political Lobbying', 'Privacy Violation', 'Legal Opinion', 'Financial Advice', 'Health Consultation', 'Government Decision')
    
    for line in csv_reader:
        index = int(line[0])
        category = categories[math.floor(index / 30)]
        csv_writer.writerow([line[0],category,line[1],line[2],str(int(line[3])^1)])
    
    csv_out.close()
    csv_in.close()