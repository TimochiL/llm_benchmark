import matplotlib.pyplot as plt
import numpy as np
import csv, math

types = (2,4,'fp16')
categories = ('Illegal Activity', 'Hate Speech', 'Malware', 'Physical Harm', 'Economic Harm', 'Fraud', 'Pornography', 'Political Lobbying', 'Privacy Violation', 'Legal Opinion', 'Financial Advice', 'Health Consultation', 'Government Decision')
folder = "llama27bchat_csv"

scores = dict()
scores_categorized = dict()
for type in types:
    scores[type] = 0
    scores_categorized[type] = dict()
    for category in categories:
        scores_categorized[type][category] = 0

for type in scores.keys():
    csv_path = f"{folder}\quant_{type}_questions_and_responses_0-389_redacted.csv"
    csv_in = open(csv_path, "r",newline="")
    csv_in.readline()
    csv_reader = csv.reader(csv_in)
    
    num_correct = 0
    
    for line in csv_reader:
        if line[4] == '1':
            line_category = line[1]
            scores_categorized[type][line_category] = scores_categorized[type][line_category] + 1
            num_correct += 1

    scores[type] = num_correct
    
    csv_in.close()

# Build Totals Visuals

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

# plt.show()

# Build Categories Visuals

bar_width = 0.06
fig, ax = plt.subplots(figsize=(16, 9))

br_lists = []
colors = ('#2f4f4f','#8b4513','#008000','#4b0082','#ff0000','#ffff00','#00ff00','#00ffff','#0000ff','#ff00ff','#6495ed','#f5deb3','#ff69b4')

category_index = 0
for category in categories:
    category_score_list = []
    for type in types:
        category_score_list.append(scores_categorized[type][category])
    # print(category_score_list)
    br = list(np.arange(len(category_score_list)))
    if category_index > 0:
        br = [x + bar_width for x in br_lists[category_index-1]]
    br_lists.append(br)
    
    plt.bar(br, category_score_list, color=colors[category_index], width=bar_width, edgecolor='grey',label=categories[category_index])
    
    category_index += 1
    
plt.xticks([r + math.floor(len(categories)/2)*bar_width for r in range(len(types))], [(str(i) + '-bit') if isinstance(i,int) else i.upper() for i in types])

ax.invert_xaxis()

ax.set_title('Forbidden Prompts Failed in Each Quant Type and Category',pad=0.5,y=1.07)

plt.legend()

# Build Categories Visuals

br_lists = []
# colors = ('#2f4f4f','#8b4513','#008000','#4b0082','#ff0000','#ffff00','#00ff00','#00ffff','#0000ff','#ff00ff','#6495ed','#f5deb3','#ff69b4')

for type in types:
    fig, ax = plt.subplots(figsize=(9, 9))
    category_score_list = []
    num_incorrect = (390 - scores[type])
    
    for category in categories:
        category_score_list.append((30-scores_categorized[type][category])/num_incorrect*100)
    trimmed_labels = []
    for i in range(len(categories)):
        if category_score_list[i] != 0:
            trimmed_labels.append(categories[i])
    
    ax.pie(list(filter(lambda a: a != 0, category_score_list)), labels=trimmed_labels, autopct='%.0f%%', pctdistance=0.9, labeldistance=1.1, textprops={'size': 'smaller'}, radius=1)
    ax.set_title(f'Forbidden Prompts Quant {(str(type) + '-bit') if isinstance(type,int) else type.upper()} Responses Fail Biases in Each Category',pad=0.5,y=1.07)

plt.show()