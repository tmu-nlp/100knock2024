with open("popular-names.txt", "r") as f:
    val = int(input("Enter the number of lines you want to display: "))
    lines = f.readlines()
    for i in range(max(0, len(lines) - val), len(lines)):
        print(lines[i].strip())
