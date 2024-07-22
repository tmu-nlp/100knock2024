# task 65. アナロジータスクでの正解率
# 64の実行結果を用い，意味的アナロジー（semantic analogy）と 
# 文法的アナロジー（syntactic analogy）の正解率を測定せよ

with open("output/ch7/knock64_out.txt", "r") as txt:
    curr_sec = None
    total_syn = 0
    match_syn = 0
    total_sem = 0
    match_sem = 0
    
    for line in txt:
        line = line.strip()
        if line.startswith(':'):
            current_section = line[1:].strip()
        else:
            parts = line.split('|')
            if len(parts) == 2:
                before_token = parts[0].strip().split()[-1]
                after_token = parts[1].strip().split()[0]
                if current_section and current_section.startswith('gram'):
                    total_syn += 1
                    if before_token.lower() == after_token.lower():
                        match_syn += 1
                else:
                    total_sem += 1
                    if before_token.lower() == after_token.lower():
                        match_sem += 1 # viet_nam?

acc_syn = (match_syn / total_syn * 100) if total_syn > 0 else 0
acc_sem = (match_sem / total_sem * 100) if total_sem > 0 else 0

print(f'Semantic Accuracy: {acc_sem:.2f}%')
print(f'Syntactic Accuracy: {acc_syn:.2f}%')

'''
Semantic Accuracy: 73.09%
Syntactic Accuracy: 74.01%
'''