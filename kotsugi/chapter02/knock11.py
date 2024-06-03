with open("./woodnx/chapter02/popular-names.txt", "r") as rf:
  for line in rf:
    print(line.strip().replace('\t', " "))

# sed '' 'woodnx/chapter02/popular-names.txt' | tr '\t' ' ' | less
