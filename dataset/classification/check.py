# Start Generation Here
with open('IAT__interior.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()[:50001]  # 读取前50000行

# 检查重复行
seen = set()
duplicates = set()

for line in lines:
    line = line.strip()  # 去除换行符
    if line in seen:
        duplicates.add(line)  # 如果已经见过，添加到重复集合
    else:
        seen.add(line)  # 否则，添加到已见集合

# 输出重复行
if duplicates:
    print("发现重复的行:")
    for duplicate in duplicates:
        print(duplicate)
else:
    print("没有发现重复的行。")
# End Generation Here
