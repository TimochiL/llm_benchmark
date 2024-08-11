import matplotlib.pyplot as plt
import csv, math

types = (2,4,'fp16')
folder = "llama27bchat_csv"

scores = dict()
for type in types:
    scores[type] = 0

for type in scores.keys():
    csv_path = f"{folder}\quant_{type}_questions_and_responses_0-389_redacted.csv"
    csv_in = open(csv_path, "r",newline="")
    csv_in.readline()
    csv_reader = csv.reader(csv_in)
    
    num_correct = 0
    for line in csv_reader:
        if line[4] == '1':
            num_correct += 1

    scores[type] = num_correct
    
    csv_in.close()

# Build visuals

quant_types = list( (str(i) + '-bit') if isinstance(i,int) else i.upper() for i in scores.keys())
quant_scores = list(scores.values())

fig, ax = plt.subplots(figsize=(6.5, 6))

plt.bar(quant_types, quant_scores)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
    
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

ax.xaxis.set_tick_params(pad = 10)
ax.yaxis.set_tick_params(pad = 5)

ax.invert_xaxis()

for i in ax.patches:
    plt.text(i.get_x()+0.8, i.get_height()+5, 
        f"{i.get_height()} of 390\n{round(i.get_height() / 3.90, 2)}% Passed",
        fontsize = 10, fontweight ='bold',
        color ='grey')

ax.set_title('Key-Value (KV) Cache Quantization Forbidden Prompts Evaluation',pad=0.5,y=1.07)

plt.show()